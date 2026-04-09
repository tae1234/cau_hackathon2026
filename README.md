# cau_hackathon2026

# 본선은 예선보다 난이도가 급상승할 수 있습니다. 2인의 팀이 참가하는 것보다 4명이 팀을 꾸려 참가하시는것을 권장드립니다.

# 예선에 참가하시는동안 팀원을 증원하셔도 괜찮으니 언제든지 mediai@cau.ac.kr 로 연락주시길 바랍니다

# 주말에는 리더보드를 업데이트하지 않습니다! 주말동안 올린 제출물들은 월요일에 업데이트 합니다

## 📝주제: 의료 영상의 다각도 인지: 퓨샷 도메인 일반화 챌린지 (Multi-View Perception in Medical Imaging: Few-Shot Domain Generalization Challenge) Hackathon Guide
## 본 파일은 해커톤 참가를 위한 모델 추론(Inference) 및 제출 파일 생성 가이드입니다. 

참가자는 제공된 inference.py를 수정하여 최종 모델 가중치와 함께 제출해야 합니다.

### 1. 프로젝트 구조제출 시 파일 구조는 아래와 같아야 합니다.

/submission

├── inference.py          # 모델 정의 및 추론 로직이 포함된 코드

├── model.pth             # 학습 완료된 모델 가중치 파일 (pt 또는 pth)

├── requirements.txt      # (선택) 추가 사용 라이브러리 목록

└── log.txt               # 부정행위 방지용 로그파일

#### 추가적인 파일을 업로드하셔도 됩니다. import하시는걸 잊지 마세요.

---

### 2. 데이터셋 정보 (Evaluation)평가 시 운영진의 데이터 환경은 다음과 같습니다.

Input Directory: /images

Image Format: GrayScale (1-channel)

Classes: 총 11개 클래스

---

### 3. 코드 수정 가이드 (inference.py)참가자는 배포된 inference.py 내의 **[참가자 작성 구간]**을 반드시 본인의 환경에 맞게 수정해야 합니다.

① 모델 아키텍처 정의 **class MyModel(nn.Module):** 부분에 학습 시 사용한 모델의 구조를 그대로 복사하여 붙여넣으세요. 외부 라이브러리(예: timm)를 사용했다면 해당 호출 코드를 넣어야 합니다.

② 가중치 로드 로직torch.save 방식에 따라 state_dict를 불러오는 방식이 다를 수 있습니다.**checkpoint['net']** 형태인지, 아니면 바로 checkpoint 자체가 가중치인지 확인하여 로드 로직을 수정하세요.

③ 전처리 (Transforms)제공된 코드의 기본 전처리는 다음과 같습니다. 학습 시 다른 전처리를 사용했다면 이 부분을 수정하십시오.

Pythontransforms.Compose([  
&nbsp;&nbsp;&nbsp;&nbsp;transforms.ToTensor(),  
&nbsp;&nbsp;&nbsp;&nbsp;transforms.Normalize(mean=[0.5], std=[0.5])     
])

---

### 4. 제출 파일 형식 (submission.csv)
본 해커톤은 **Macro F1-Score**를 기준으로 채점하며, 결과 파일은 반드시 **원핫 인코딩(One-hot Encoding)** 형식을 준수해야 합니다.

Index&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bladder&nbsp;&nbsp;&nbsp;&nbsp;femur-left&nbsp;&nbsp;&nbsp;&nbsp;femur-right&nbsp;&nbsp;&nbsp;...&nbsp;&nbsp;&nbsp;spleen

image_00000&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...&nbsp;&nbsp;&nbsp;0

image_00001&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...&nbsp;&nbsp;&nbsp;0

**Index**: 확장자를 포함하지 않은 이미지 파일명

**bladder~spleen**: 모델이 예측한 결과 (해당되는 클래스에만 '1', 나머지는 '0')

**컬럼명 순서**: bladder, femur-left, femur-right, heart, kidney-left, kidney-right, liver, lung-left, lung-right, pancreas, spleen

---

### 5. 실행 방법운영진은 아래 명령어를 통해 제출하신 코드를 검증합니다.

>python inference.py --input_dir ./images --weight_path ./model.pth --output_csv ./submission.csv

⚠️ **주의사항** 재현성: 제출한 가중치와 코드로 생성한 submission.csv가 실제 채점 결과와 다를 경우 불이익이 있을 수 있습니다.

라이브러리: 표준 라이브러리(PyTorch, Torchvision, Pandas, Pillow) 외의 특수한 패키지를 사용한 경우 반드시 **requirements.txt**를 동봉하세요.

오류 처리: 이미지 로딩 실패 시 해당 이미지는 skip 하도록 구성되어 있으나, 가급적 모든 테스트 데이터에 대해 결과가 도출되도록 확인 바랍니다.

---

### 6. 모델 검증 및 부정행위 관련 규정

🚫 제한 조건 (Strict Constraints - Zero Tolerance) 

#### 외부 데이터 사용 전면 금지: 참가자는 오직 주최측에서 제공한 공식 학습 데이터셋만을 사용해야 합니다.

#### 사전 학습 가중치 사용 불가: ImageNet 등으로 사전 학습된 가중치(Pre-trained weights) 사용은 허용되지 않습니다. 반드시 무작위 초기화(Random Initialization) 상태에서 제공된 의료 영상 데이터만으로 학습해야 합니다.

#### 허용되는 조건: Model Merge 기법, 데이터 증강같은 기법으로 추가학습은 가능합니다.

#### 적발 시 조치: 대회 진행 및 코드 리뷰 과정에서 위반 사항 발견 시, 사전 경고 없이 즉각 탈락(Disqualification) 처리됩니다.

필수 제출 의무: <span style="color:red">모든 참가자는 최종 모델 제출 시 배포된 train 폴더와 가중치 파일(.pth/.pt 등)의 고유 해시값(Hash) 및 학습 진행 로그(Log)를 함께 제출해야 합니다.

재학습 검증(Retraining Verification): <span style="color:red">주최 측은 모델의 무결성 검증을 위해 참가자에게 전처리 및 학습 코드 전체를 요구할 수 있습니다.

실격 규정: <span style="color:red">제출된 코드를 주최 측 환경에서 재학습했을 때 성능이 재현되지 않거나, 제출된 로그/해시값이 조작된 것으로 확인될 경우 사전 경고 없이 실격 처리됩니다.

train 폴더와 가중치 파일의 고유 해시값 체크 방법

파일경로 부분을 train폴더와 가중치파일 주소로 넣고 실행하면 hash값이 생성됩니다. 그 해시값을 log에 남겨주세요.

```python
import hashlib
import os

def get_official_hash(folder_path):
    file_hashes = []
    for root, dirs, files in os.walk(folder_path):
        for file in sorted(files):
            if file.startswith('.'): continue

            file_path = os.path.join(root, file)

            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            file_hashes.append(hasher.hexdigest().upper())

    combined = "".join(file_hashes)
    final_hash = hashlib.md5(combined.encode()).hexdigest().upper()
    return final_hash

print(get_official_hash('train 폴더경로'))
print(get_official_hash('가중치 파일경로'))
```

---

# 행운을 빕니다!
