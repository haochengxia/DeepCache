import os
import sys
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.inception import inception_v3
from tqdm import tqdm
import numpy as np
from scipy.stats import entropy

def calculate_inception_score(images, batch_size=16, splits=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(images)
    assert N > batch_size, "生成的图像数量应大于批次大小"

    # 定义图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Inception V3 的均值和标准差
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 加载 Inception V3 模型
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # 获取预测结果
    preds = np.zeros((N, 1000))

    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size)):
            batch_images = images[i:i+batch_size]
            batch = torch.stack([preprocess(img) for img in batch_images], dim=0).to(device)

            pred = inception_model(batch)
            preds[i:i+batch.shape[0]] = F.softmax(pred, dim=1).cpu().numpy()

    # 计算 Inception Score
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    is_mean = np.mean(split_scores)
    is_std = np.std(split_scores)
    return is_mean, is_std

def main():
    if len(sys.argv) < 2:
        print("用法: python inception_score.py <generated_images.pt>")
        sys.exit(1)

    generated_images_path = sys.argv[1]

    if not os.path.isfile(generated_images_path):
        print(f"[ERROR] {generated_images_path} 不是有效的 .pt 文件")
        sys.exit(1)

    print("加载生成的图像...")
    ckpt = torch.load(generated_images_path)
    all_images = [transforms.ToPILImage()(i) for i in ckpt['images']]

    print("计算 Inception Score...")
    is_mean, is_std = calculate_inception_score(all_images, batch_size=16, splits=10)
    print(f"Inception Score: {is_mean} ± {is_std}")

if __name__ == '__main__':
    main()

