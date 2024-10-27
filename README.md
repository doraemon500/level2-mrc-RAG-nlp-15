<div align='center'>

  # 🏆 LV.2 NLP 프로젝트 : Open-Domain Question Answering

</div>

## ✏️ 대회 소개

| 특징     | 설명 |
|:------:| --- |
| 대회 주제 | 네이버 부스트캠프 AI Tech 7기 NLP Track의 Level 2 도메인 기초 대회 'Open-Domain Question Answering (Machine Reading Comprehension)'입니다. |
| 대회 설명 | 주어지는 Documents의 내용을 기반으로 질문이 주어지면, 그 질문에 대한 정확한 답변을 문서에서 찾아내는 것을 목표로 합니다. |
| 데이터 구성 | 데이터는 위키피디아의 내용으로 대부분 이루어진 문서 데이터, 그리고 Question과 Answer로 구성되어 있습니다. |
| 평가 지표 | 답변을 정확히 추출하는지를 확인하기 위해 EM(Exact Match) 지표가 사용되었습니다.|


## 🎖️ Leader Board
프로젝트 결과 Public 리더보드 2등, Private 리더보드 2등 (공동 EM Score)을 기록하였습니다.
###  Public Leader Board 
![leaderboard_mid](./docs/leaderboard_mid.png)

###  Private Leader Board 
![leaderboard_final](./docs/leaderboard_final.png)

## 👨‍💻 15조가십오조 멤버
<div align='center'>
  
| 김진재 [<img src="./docs/github_official_logo.png" width=20 style="vertical-align:middle;" />](https://github.com/jin-jae) | 박규태 [<img src="./docs/github_official_logo.png" width=20 style="vertical-align:middle;" />](https://github.com/doraemon500) | 윤선웅 [<img src="./docs/github_official_logo.png" width=20 style="vertical-align:middle;" />](https://github.com/ssunbear) | 이정민 [<img src="./docs/github_official_logo.png" width=20 style="vertical-align:middle;" />](https://github.com/simigami) | 임한택 [<img src="./docs/github_official_logo.png" width=20 style="vertical-align:middle;" />](https://github.com/LHANTAEK)
|:-:|:-:|:-:|:-:|:-:|
| ![김진재](https://avatars.githubusercontent.com/u/97018331) | ![박규태](https://avatars.githubusercontent.com/u/64678476) | ![윤선웅](https://avatars.githubusercontent.com/u/117508164) | ![이정민](https://avatars.githubusercontent.com/u/46891822) | ![임한택](https://avatars.githubusercontent.com/u/143519383) |

</div>

  
## 👼 역할 분담
<div align='center'>

|팀원   | 역할 |
|------| --- |
| 김진재 |  |
| 박규태 |  |
| 윤선웅 |  |
| 이정민 |  |
| 임한택 |  |

</div>


## 🏃 프로젝트
### 🖥️ 프로젝트 개요
|개요| 설명 |
|:------:| --- |
| 주제 |  |
| 목표 |  |
| 평가 지표 |  |
| 개발 환경 |  |
| 협업 환경 |  |

### 📅 프로젝트 타임라인
- 프로젝트는 2024-09-30 ~ 2024-10-25까지 진행되었습니다.

<div align='center'>


</div>

### 🕵️ 프로젝트 진행
- 프로젝트를 진행하며 단계별로 실험하여 적용한 내용들을 아래와 같습니다.


|  프로세스   | 설명 |
|:-------:| --- |
| EDA     |  |
| 전처리    |  |
| 증강     |  |
| 모델 선정 |  |
| 앙상블    |  |

### 📊 Dataset
- TBD

### 🤖 Ensemble Model

| Model | loss | learning_rate| batch_size | Used Dataset |
| ------------------------- | --- | --- | ----- | --- |


## 📁 프로젝트 구조
프로젝트 폴더 구조는 다음과 같습니다.
```
.
├── data
│   ├── test_dataset
│   ├── train_dataset
│   └── wikipedia_documents.json
├── models
├── output
├── README.md
├── requirements.txt
├── run.py
└── src
    ├── arguments.py
    ├── CNN_layer_model.py
    ├── data_analysis.py
    ├── ensemble
    │   ├── probs_voting_ensemble_n.py
    │   ├── probs_voting_ensemble.py
    │   └── scores_voting_ensemble.py
    ├── main.py
    ├── optimize_retriever.py
    ├── preprocess_answer.ipynb
    ├── qa_trainer.py
    ├── retrieval_2s_rerank.py
    ├── retrieval_BM25.py
    ├── retrieval_Dense.py
    ├── retrieval_hybridsearch.py
    ├── retrieval.py
    ├── retrieval_SPLADE.py
    ├── retrieval_tfidf.py
    └── utils.py
```

### 📦 src 폴더 구조 설명
- TBD

### 📁 보충 설명
- TBD

### 💾 Installation
- `python=3.10` 환경에서 requirements.txt를 pip로 install 합니다. (```pip install -r requirements.txt```)
- `python run.py`를 입력하여 프로그램을 실행합니다.