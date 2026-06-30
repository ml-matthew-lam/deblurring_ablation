$$ \mathrm{SCA}(\mathbf x) = \mathbf x * W \, \text{global_avg_pool}(\mathbf x) $$
$$ \mathrm {SimpleGate} (\mathbf x) = \mathbf x_1 \odot \mathbf x_2$$
$$\mathrm{GELU}(x) = x\Phi(x), \quad \text{where} \quad \Phi(x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi}} e^{-x^2/2}$$
$$ \mathrm{SSIM} = \frac {(2\mu_x \mu_y+c_1)(2\sigma_{xy}+c_2)} {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)} $$

To implement SSIM using convolutions, we consider the values in the $x$ and $y$ patches to be values of a random variable $X$ and $Y$ respectively, such that $\mu_x = E(X)$, $\sigma_x^2 = \mathrm{Var}(X)$, and $\sigma_{xy}=\mathrm{Cov}$. We then use the following facts: 
- $E(X)= x* w$, where $w$ is a kernel with weights summing to $1$
- $\mathrm{Var}(X) = E(X^2)-[E(X)]^2$
- $\mathrm{Cov}(X,Y)=E(XY)-E(X)E(Y)$

Thus, we can follow these steps to calculate the average SSIM across all channels and patches centered at each pixel. The predicted tensor $\mathbf x$ and ground truth tensor $\mathbf y$ have dimensions $[B,C,H,W]$.
- Apply a depthwise convolution ($\sum \mathrm{weights} = 1$) to the predicted tensor. Do the same for the ground truth. This produces maps of $\mu_x$ and $\mu_y$ values for each of the channels. We will call the resulting tensors $[\mu_x]$ and $[\mu_y]$, each of dimensions $[B,C,H,W]$.
- Apply the same depthwise convolution to $\mathbf x \odot \mathbf x$ and $\mathbf y \odot \mathbf y$, then subtract $[\mu_x^2]$ and $[\mu_y^2]$ respectively. This produces tensors $[\sigma_x^2]$ and $[\sigma_y^2]$, each of dimensions $[B,C,H,W]$.
- Apply the same depthwise convolution to $\mathbf x \odot \mathbf y$, then subtract $[\mu_x]\odot [\mu_y]$. This produces the tensor $[\sigma_{xy}]$.
- Apply the SSIM formula elementwise to obtain $[\mathrm{SSIM}]$.
- Take the average across the pixel dimensions in $[\mathrm{SSIM}]$, then take the average across the channel dimension, then across the batch. This will produce the (scalar) average SSIM value. Then $\mathcal L_\mathrm{SSIM} = 1-\mathrm{SSIM}$.

$$\mathcal L = \alpha \mathcal L_1 + (1-\alpha)\mathcal L_\mathrm{SSIM}$$
