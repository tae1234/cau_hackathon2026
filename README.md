# cau_hackathon2026

## 📝주제: 의료 영상의 다각도 인지: 퓨샷 도메인 일반화 챌린지 (Multi-View Perception in Medical Imaging: Few-Shot Domain Generalization Challenge) Hackathon Guide
## 본 파일은 해커톤 참가를 위한 모델 추론(Inference) 및 제출 파일 생성 가이드입니다. 

참가자는 제공된 inference.py를 수정하여 최종 모델 가중치와 함께 제출해야 합니다.

### 1. 프로젝트 구조제출 시 파일 구조는 아래와 같아야 합니다.

/submission

├── inference.py          # 모델 정의 및 추론 로직이 포함된 코드

├── model.pth             # 학습 완료된 모델 가중치 파일 (pt 또는 pth)

└── requirements.txt      # (선택) 추가 사용 라이브러리 목록

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

### 4. 제출 파일 형식 (submission.csv)inference.py 실행 시 생성되는 submission.csv는 반드시 아래의 컬럼 구조를 가져야 합니다. (AUC 평가를 위해 클래스별 확률값이 포함되어야 합니다.)

filename&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;predicted_class&nbsp;&nbsp;&nbsp;prob_0&nbsp;&nbsp;&nbsp;&nbsp;prob_1&nbsp;&nbsp;&nbsp;...&nbsp;&nbsp;&nbsp;prob_10

image_000.png&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.01&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0.02&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;...&nbsp;&nbsp;&nbsp;0.05

**filename**: 확장자를 포함한 이미지 파일명 (정렬 필수)

**predicted_class**: 모델이 예측한 최종 클래스 인덱스 (0~10)

**prob_n**: 해당 클래스일 확률 (Softmax 결과값, 0.0 ~ 1.0)


### 5. 실행 방법운영진은 아래 명령어를 통해 제출하신 코드를 검증합니다.

>python inference.py --input_dir ./images --weight_path ./model.pth --output_csv ./submission.csv

⚠️ **주의사항** 재현성: 제출한 가중치와 코드로 생성한 submission.csv가 실제 채점 결과와 다를 경우 불이익이 있을 수 있습니다.

라이브러리: 표준 라이브러리(PyTorch, Torchvision, Pandas, Pillow) 외의 특수한 패키지를 사용한 경우 반드시 **requirements.txt**를 동봉하세요.

오류 처리: 이미지 로딩 실패 시 해당 이미지는 skip 하도록 구성되어 있으나, 가급적 모든 테스트 데이터에 대해 결과가 도출되도록 확인 바랍니다.
