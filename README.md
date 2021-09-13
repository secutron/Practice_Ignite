# [KR] Practice_Ignite

[Torch Ignite](https://pytorch.org/ignite/)를 이용한 분산 딥러닝(Distributed Deep Learning - DDL) 가이드라인 입니다.

## [Ignite](https://pytorch.org/ignite/) 필요성

 연구 주제에 따라, 학습시켜야 하는 모델의 크기나 데이터의 양이 매우 커지는 경우가 있습니다. 이러한 경우 우리는, 다수의 컴퓨터에 분산되어 있는 복수의 GPU에서 대량의 데이터 또는 거대 모델의 학습을 가능하게 해주는 DDL 기술의 도움을 받을 수 있습니다. 

 다만 이러한 DDL 기술 자체를 어렵게 느끼는 경우도 많은데, 그 이유 중 하나는 DDL 구현 방식이 매우 다양하다는 점에 있다고 생각합니다. 예를 들어 DDL은, RPC(Remote Procedure Call)를 직접 이용하거나 파이썬의 multiprocessing 구현체를 이용하여 구현할 수도 있습니다. 이 외에도 매우 많은 종류의 접근방식이 존재하며, 이들 간의 상호 조합 역시 가능합니다. 따라서 처음 DDL을 시도하는 경우, 단편화되어 있는 웹 상의 정보로부터 혼란을 겪는 경우가 일반적입니다. 

 이에 반해 이그나이트 프레임워크는, 딥러닝에서 자주 사용되는 DDL 방식을 최소의 코드량으로 구현할 수 있도록 체계화 ---~~라고 쓰고 강제화라고 읽음~~--- 하고 있습니다. 따라서 분산처리(distributed processing) 개념에 익숙하지 않은 연구자도 DDL에 쉽고 빠르게 접근할 수 있도록 도움을 줍니다.
 

## [ignite](https://pytorch.org/ignite/)가 DDL을 위한 유일한 솔루션인가요? 

 아닙니다. 같은 파이토치 기반의 래퍼 프레임워크인 라이트닝(lightning - 라이트닝 개발자가 [이그나이트와 fast.ai를 비교한 글](https://towardsdatascience.com/pytorch-lightning-vs-pytorch-ignite-vs-fast-ai-61dc7480ad8a) 참조), 텐서플로 (tensorflow) 기반의  메시 텐서플로(mesh tensorflow), 여러 백엔드(backend)에서 사용이 가능한 호로보드 (horovod) 등도 있습니다. 또는 빌트인 패러럴 서포트(built-in parallel support)인 torch.distributed이나 tf.distribute 등을 직접 이용해도 됩니다. ---~~읭? 스펠이 다르다.~~---


## [ignite](https://pytorch.org/ignite/) 한계

 안타깝게도 현 시점의 ignite는 데이터 패러럴리즘만 지원하는 것으로 보입니다. ---~~물론 하위 모듈을 이용한 모델 패러럴리즘 구현은 가능~~---
 
 
<div align="center">
<img width=1024 src="https://i.imgur.com/YrO6THF.png"/>
</div>

 이해를 돕기 위해 데이터 패러럴리즘을 간략히 설명하면 다음과 같습니다. 먼저 상기 그림 좌측의 데이터 패러럴리즘 항목에서 각 GPU 0번과 GPU 1번이 준비되어 있음을 알 수 있습니다. 그리고 학습 데이터로 i0, i1, i2, i3과 같이 4개 샘플이 준비되어 있는데, 이는 GPU 숫자와 같은 2개의 그룹으로 나뉘어 각 GPU로 전달되고 있습니다. 즉 GPU 0번에는 i0, i1 샘플이, 그리고 GPU 1번에는 i2, i3 샘플이 전달됩니다. 그리고 이렇게 나뉘어진 샘플은 모델로 전달되고 각각의 출력인 o0, 01과 o2, o3가 다시 한 곳으로 모여 손실값으로 계산되어집니다.  이때 모델은 ‘동일한 모델이 각 GPU에 복제’되어 있는 형태이며, 따라서 그림에서는 동일한 색상의 ‘m’ 으로 표시하였습니다. 

 여러 분산 딥러닝 방식 중에서 데이터 패러럴리즘이 가장 이해하기 쉽습니다. 그리고 사실 학습데이터의 양에 상관 없이 항상 데이터는 나누어 처리될 수 있으므로, DDL에서 가장 많이 사용하는 형태이라고 생각할 수 있습니다. 


## Repo. 구성

- 2.1.1.ipynb : Colab VM의 CPU를 이용한 간단 분산처리
- 2.1.2.ipynb : Colab VM의 GPU를 이용한 간단 분산처리
- 2.1.3.ipynb : Colab VM의 TPU를 이용한 간단 분산처리
- 2.1.4.ipynb : 간단 분산 딥러닝 예제
- A.1.ipynb : 간단 속성 예제
- A.1.x1.ipynb : auto_dataloader에 의한 데이터 분배 확인 예제
- A.1.x2.ipynb : 기타
---~~간단간단간단간단~~--


## **Disclosure**:   
Most of the code in this repo is inspired by [ignite](https://pytorch.org/ignite/) team's great works. You can find it here: https://pytorch.org/ignite/master/about.html . I appreciate their contribution to the deep learning commnunity!
