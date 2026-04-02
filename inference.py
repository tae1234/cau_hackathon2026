import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ==========================================
# [참가자 작성 구간] 모델 정의
# 본인이 학습에 사용한 모델 클래스 구조를 여기에 정의하거나 import 하세요.
# ==========================================

class MyModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=11):
        super(MyModel, self).__init__()
        # 예시: 참가자가 작성할 공간
        # self.network = ...
        pass

    def forward(self, x):
        # return self.network(x)
        pass


# ==========================================
# 데이터셋 정의 (수정 불필요)
# ==========================================

class TestDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir

        self.img_names = sorted([f for f in os.listdir(img_dir)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        # MedMNIST 규격에 맞춘 GrayScale 로드
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        return img_name, image


# ==========================================
# 메인 실행 함수
# ==========================================

def get_args():
    parser = argparse.ArgumentParser(description="Hackathon Inference Script")
    parser.add_argument('--input_dir', type=str, default='./images', help='Path to test images')
    parser.add_argument('--weight_path', type=str, default='./model.pth', help='Path to trained model weights')
    parser.add_argument('--output_csv', type=str, default='./submission.csv', help='Path to save results')
    parser.add_argument('--batch_size', type=int, default=64)
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. [참가자 작성 구간] 모델 인스턴스 생성
    # 본인의 모델 설정에 맞게 수정하세요.
    model = MyModel(in_channels=1, num_classes=11)

    # 2. 가중치 로드
    try:
        # 참가자마다 저장 방식(state_dict 혹은 dict 전체)이 다를 수 있음을 유의
        checkpoint = torch.load(args.weight_path, map_location=device)

        if isinstance(checkpoint, dict) and 'net' in checkpoint:
            model.load_state_dict(checkpoint['net'])
        else:
            model.load_state_dict(checkpoint)

        print(f"Successfully loaded weights from {args.weight_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return

    model.to(device).eval()

    # 3. 전처리 정의 (학습 시와 동일하게 유지 권장)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # 4. 데이터 로더
    if not os.path.exists(args.input_dir):
        print(f"Directory not found: {args.input_dir}")
        return

    dataset = TestDataset(args.input_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    results = []
    print(f"Starting inference on {len(dataset)} images...")

    with torch.no_grad():
        for filenames, images in loader:
            images = images.to(device)
            outputs = model(images)

            # AUC 계산을 위한 Softmax 확률값
            probs = torch.softmax(outputs, dim=1)
            # 최종 예측 클래스
            _, predicted = torch.max(outputs, 1)

            for i in range(len(filenames)):
                res = {'filename': filenames[i], 'predicted_class': predicted[i].item()}
                # 0번부터 10번 클래스까지의 확률 저장 (prob_0, prob_1, ...)
                for c in range(probs.shape[1]):
                    res[f'prob_{c}'] = probs[i, c].item()
                results.append(res)

    # 5. 결과 저장 (CSV 출력)
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"Done! Results saved to {args.output_csv}")


if __name__ == "__main__":
    main()