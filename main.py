from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# 創建輸出資料夾
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# 定義 3 個直方圖均衡化演算法
def histogram_equalization(image, output_range=(0, 255)):
    """
    改進版 Global Histogram Equalization，支援設定輸出範圍。
    output_range: tuple，指定輸出灰階範圍，預設為 (0, 255)。
    """
    grayscale = image.convert("L")
    img_array = np.array(grayscale)

    # 計算直方圖
    histogram, _ = np.histogram(img_array.flatten(), bins=256, range=[0, 256])

    # 計算累積分布函數（CDF）
    cdf = histogram.cumsum()
    cdf_normalized = (cdf - cdf.min()) / (cdf.max() - cdf.min())  # 正規化至 [0, 1]

    # 映射到指定範圍
    min_val, max_val = output_range
    cdf_scaled = cdf_normalized * (max_val - min_val) + min_val
    cdf_scaled = cdf_scaled.astype('uint8')

    # 應用映射
    equalized_image_array = cdf_scaled[img_array]

    # 轉換回影像
    equalized_image = Image.fromarray(equalized_image_array)
    return equalized_image

def piecewise_linear_equalization(image, breakpoints=[0, 85, 170, 255]):
    grayscale = image.convert("L")
    img_array = np.array(grayscale)

    new_image_array = np.zeros_like(img_array)
    for i in range(len(breakpoints) - 1):
        start, end = breakpoints[i], breakpoints[i + 1]
        mask = (img_array >= start) & (img_array < end)
        new_image_array[mask] = np.interp(
            img_array[mask], [start, end], [start, 255 * (i + 1) / (len(breakpoints) - 1)]
        )

    equalized_image = Image.fromarray(new_image_array.astype('uint8'))
    return equalized_image

def local_histogram_equalization(image, kernel_size=8):
    grayscale = image.convert("L")
    img_array = np.array(grayscale)
    padded_array = np.pad(img_array, kernel_size // 2, mode="reflect")
    new_image_array = np.zeros_like(img_array)

    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            local_region = padded_array[i:i + kernel_size, j:j + kernel_size]
            local_hist, _ = np.histogram(local_region.flatten(), bins=256, range=[0, 256])
            local_cdf = local_hist.cumsum()
            local_cdf_normalized = (local_cdf - local_cdf.min()) * 255 / (local_cdf.max() - local_cdf.min())
            local_cdf_normalized = local_cdf_normalized.astype('uint8')
            new_image_array[i, j] = local_cdf_normalized[img_array[i, j]]

    equalized_image = Image.fromarray(new_image_array)
    return equalized_image

# 創建對比直方圖保存函數
def save_comparison_histogram(original_image, processed_image, output_path):
    """
    生成並保存原圖和處理後影像的直方圖比較。
    """
    # 將影像轉換為灰階陣列
    original_array = np.array(original_image.convert("L"))
    processed_array = np.array(processed_image.convert("L"))

    # 計算直方圖
    original_hist, bins = np.histogram(original_array.flatten(), bins=256, range=[0, 256])
    processed_hist, _ = np.histogram(processed_array.flatten(), bins=256, range=[0, 256])

    # 繪製直方圖
    plt.figure()
    plt.bar(bins[:-1], original_hist, width=1, color='blue', alpha=0.6, label="Original")
    plt.bar(bins[:-1], processed_hist, width=1, color='orange', alpha=0.6, label="Processed")
    plt.title('Histogram Comparison')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    # 保存圖表
    plt.savefig(output_path)
    plt.close()

# 載入原圖並檢查是否存在
# 檢查檔案是否存在
image_path = "images/Lena.png"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Error: Image file '{image_path}' not found!")

# 載入圖片
original_image = Image.open(image_path)

# 將原圖轉換為 RGB 並保存
original_image.convert("RGB").save(f"{output_dir}/Original_Image.jpg")

# 1. Global HE 測試多個參數
global_he_params = {
    "full_range": (0, 255),
    "moderate_range": (50, 200),
    "minimal_range": (100, 180),
}
for name, param in global_he_params.items():
    global_he_image = histogram_equalization(original_image, output_range=param)
    output_image_path = f"{output_dir}/GlobalHE_{name}_{param[0]}_{param[1]}.jpg"
    global_he_image.save(output_image_path)
    save_comparison_histogram(original_image, global_he_image, output_image_path.replace(".jpg", "_result.png"))

# 2. Piecewise Linear 測試多個參數
piecewise_params_list = {
    "default": [0, 128, 192, 255],
    "strong": [0, 64, 128, 192, 255],
    "mild": [0, 150, 255],
}
for name, params in piecewise_params_list.items():
    piecewise_image = piecewise_linear_equalization(original_image, breakpoints=params)
    output_image_path = f"{output_dir}/PiecewiseLinear_{name}_{'_'.join(map(str, params))}.jpg"
    piecewise_image.save(output_image_path)
    save_comparison_histogram(original_image, piecewise_image, output_image_path.replace(".jpg", "_result.png"))

# 3. Local HE 測試多個窗口大小
local_he_kernels = {
    "small": 8,
    "moderate": 16,
    "large": 32,
}
for name, kernel_size in local_he_kernels.items():
    local_he_image = local_histogram_equalization(original_image, kernel_size=kernel_size)
    output_image_path = f"{output_dir}/LocalHE_{name}_kernel_{kernel_size}.jpg"
    local_he_image.save(output_image_path)
    save_comparison_histogram(original_image, local_he_image, output_image_path.replace(".jpg", "_result.png"))

print("All results saved!")
