import torch
import os
import random
import time
import json
import psutil
from torchvision.utils import save_image
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader

import model as deblur_model
from dataset import GoProDataset

def run_benchmark(model, test_loader, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    # track current process for system RAM usage
    process = psutil.Process(os.getpid())
    peak_memory_mb = 0.0
    
    # put model in inference mode
    model.eval()
    
    results = []
    total_time = 0.0
        
    # ----- warm-up phase to get through initialization delays before inference time is measured -----
    print("----- starting warm up -----")
    with torch.no_grad():
        # 5 dummy passes to initialize memory and cache operations
        for _ in range(5):
            dummy_input = torch.randn(1, 3, 256, 256).to(device)
            _ = model(dummy_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # ----- main inference loop -----
    with torch.no_grad(): 
        for idx, (blur_img, sharp_img) in enumerate(test_loader):
            blur_img, sharp_img = blur_img.to(device), sharp_img.to(device)
            
            start_time = time.perf_counter()
            restored_img = model(blur_img)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            end_time = time.perf_counter()
            inference_time = end_time - start_time
            total_time += inference_time
            
            # track peak RAM usage during the loop
            current_ram_mb = process.memory_info().rss / (1024 ** 2)
            if current_ram_mb > peak_memory_mb:
                peak_memory_mb = current_ram_mb
            
            # clamping values to [0,1] range
            restored_img_clamped = torch.clamp(restored_img, 0.0, 1.0)
            blur_img_clamped = torch.clamp(blur_img, 0.0, 1.0)
            
            # calculate output metrics (deblurred output vs target)
            psnr_val = psnr_metric(restored_img_clamped, sharp_img).item()
            ssim_val = ssim_metric(restored_img_clamped, sharp_img).item()
            
            # calculate input metrics (blurry input vs target)
            base_psnr = psnr_metric(blur_img_clamped, sharp_img).item()
            base_ssim = ssim_metric(blur_img_clamped, sharp_img).item()
            
            psnr_improvement = psnr_val - base_psnr
            ssim_improvement = ssim_val - base_ssim

            results.append({
                'index': idx,
                'psnr': psnr_val,
                'ssim': ssim_val,
                'base_psnr': base_psnr,
                'base_ssim': base_ssim,
                'psnr_improvement': psnr_improvement,
                'ssim_improvement': ssim_improvement,
                'time': inference_time,
                'blur_tensor': blur_img.cpu(),      
                'restored_tensor': restored_img_clamped.cpu(),
                'sharp_tensor': sharp_img.cpu()
            })
            
            if idx != 0 and idx%100 == 0:
                print(f"processed {idx} of {len(test_loader)} images")

    # calculate averages
    avg_psnr = sum(r['psnr'] for r in results) / len(results)
    avg_ssim = sum(r['ssim'] for r in results) / len(results)
    
    avg_base_psnr = sum(r['base_psnr'] for r in results) / len(results)
    avg_base_ssim = sum(r['base_ssim'] for r in results) / len(results)
    
    avg_psnr_improvement = sum(r['psnr_improvement'] for r in results) / len(results)
    avg_ssim_improvement = sum(r['ssim_improvement'] for r in results) / len(results)

    avg_time = total_time / len(results)

    # sorting
    results_by_psnr = sorted(results, key=lambda x: x['psnr'])
    results_by_ssim = sorted(results, key=lambda x: x['ssim'])
    results_by_psnr_imp = sorted(results, key=lambda x: x['psnr_improvement'])
    results_by_ssim_imp = sorted(results, key=lambda x: x['ssim_improvement'])

    worst_4_psnr = results_by_psnr[:4]
    best_4_psnr = results_by_psnr[-4:]
    worst_4_ssim = results_by_ssim[:4]
    best_4_ssim = results_by_ssim[-4:]
    
    least_improved_4_psnr = results_by_psnr_imp[:4]
    most_improved_4_psnr = results_by_psnr_imp[-4:]
    least_improved_4_ssim = results_by_ssim_imp[:4]
    most_improved_4_ssim = results_by_ssim_imp[-4:]

    # ensuring deterministic selection
    seeded_rng = random.Random(42)
    random_10 = seeded_rng.sample(results, 10)

    def save_trio(item, category, rank=None):
        # concatenate images
        grid = torch.cat((item['blur_tensor'][0], item['restored_tensor'][0], item['sharp_tensor'][0]), dim=2)
        
        rank_str = f"_rank{rank}" if rank is not None else ""
        base_filename = f"{category}{rank_str}_idx{item['index']}"
        
        save_image(grid, os.path.join(output_dir, f"{base_filename}.png"))

        # create sub-folder for the triplet
        sub_folder_path = os.path.join(output_dir, base_filename)
        os.makedirs(sub_folder_path, exist_ok=True)

        # save individual images with metrics in the filenames
        blur_name = f"blurry_psnr{item['base_psnr']:.2f}_ssim{item['base_ssim']:.4f}.png"
        save_image(item['blur_tensor'][0], os.path.join(sub_folder_path, blur_name))
        
        restored_name = f"restored_psnr{item['psnr']:.2f}_ssim{item['ssim']:.4f}.png"
        save_image(item['restored_tensor'][0], os.path.join(sub_folder_path, restored_name))
        
        sharp_name = "sharp_target.png"
        save_image(item['sharp_tensor'][0], os.path.join(sub_folder_path, sharp_name))

    # saving images in two formats as defined by save_trio
    for i, item in enumerate(worst_4_psnr): save_trio(item, "worst_psnr", i+1)
    for i, item in enumerate(best_4_psnr): save_trio(item, "best_psnr", 4-i)
    for i, item in enumerate(worst_4_ssim): save_trio(item, "worst_ssim", i+1)
    for i, item in enumerate(best_4_ssim): save_trio(item, "best_ssim", 4-i)
    
    for i, item in enumerate(least_improved_4_psnr): save_trio(item, "least_improved_psnr", i+1)
    for i, item in enumerate(most_improved_4_psnr): save_trio(item, "most_improved_psnr", 4-i)
    for i, item in enumerate(least_improved_4_ssim): save_trio(item, "least_improved_ssim", i+1)
    for i, item in enumerate(most_improved_4_ssim): save_trio(item, "most_improved_ssim", 4-i)

    for item in random_10: save_trio(item, "random_selection")

    # saving metrics
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report = {
        "model_parameters": total_params,
        "average_input_psnr": avg_base_psnr,
        "average_output_psnr": avg_psnr,
        "average_psnr_improvement": avg_psnr_improvement,
        "average_input_ssim": avg_base_ssim,
        "average_output_ssim": avg_ssim,
        "average_ssim_improvement": avg_ssim_improvement,
        "average_inference_time_seconds": avg_time,
        "peak_system_ram_usage_mb": peak_memory_mb,
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"----- DONE -----")
    return report

if __name__ == "__main__":
    # ///////////  CONFIGURATIONS  ///////////////
    # specify what type of model / block type is being tested below:
    block_type = ["NAF", "GELU"][1]
    # specify the filepath to the saved trained weights below:
    weights_path = "./NAF_best.pth"
    # specify the name of the folder where the results will be saved:
    output_folder = "NAF_results"
    # ///////////////////////////////////////////

   
    # set hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"initializing benchmark on: {device}")
    model = deblur_model.UNet(deblur_model.NAFBlock if block_type=="NAF" else deblur_model.GELUBlock, 32)
    
    # load weights from training
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    test_dataset = GoProDataset("./GOPRO_Large", 256, "test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    run_benchmark(model=model, test_loader=test_loader, device=device, output_dir=output_folder)