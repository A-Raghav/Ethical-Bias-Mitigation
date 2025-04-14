# FairAI
FairAI is a wrapper around the open source AI-Fairness-360 library (by IBM), that makes it easier to implement detection and mitigation of bias in ML models throughout the lifecycle of an project in both dev and prod settings.

FairAI -
1. Checks for bias in input data and predictions of ML models
2. Mitigates the bias in ML model's predictions

The tutorial notebooks in [fairai.notebooks.examples](https://gitlab.innovaccer.com/aseem.raghav/bias-fairness-framework/-/tree/master/notebooks/examples) provide a demonstration on how one can easily integrate a bias mitigation mechanism in their ML project. 

To utilise the full power and customisability of bias-fairness framework, users are advised to also refer to the [AIF-360 library by IBM](https://github.com/Trusted-AI/AIF360).

---

## Supported bias mitigation algorithms
* Reweighing ([Kamiran and Calders, 2012](http://doi.org/10.1007/s10115-011-0463-8))
* Reject Option Classification ([Kamiran et al., 2012](https://doi.org/10.1109/ICDM.2012.45))
* Equalized Odds Postprocessing ([Hardt et al., 2016](https://papers.nips.cc/paper/6374-equality-of-opportunity-in-supervised-learning))
* Calibrated Equalized Odds Postprocessing ([Pleiss et al., 2017](https://papers.nips.cc/paper/7151-on-fairness-and-calibration))

---

## Installing FairAI from Git
```pip install git+https://gitlab.innovaccer.com/aseem.raghav/bias-fairness-framework.git```

## Installing FairAI manually
Clone the latest version of this repository -

```git clone https://gitlab.innovaccer.com/aseem.raghav/bias-fairness-framework.git```

Navigate to the root directory of the project and run -

```pip install .```


# Ethical-Bias-Mitigation
A Python toolkit for detecting and mitigating ethical bias in machine learning models. This project provides wrapper classes around IBM's AI Fairness 360 (AIF360) library to make bias detection and mitigation more accessible and easier to implement in machine learning pipelines.

### About
This repository contains tools to help data scientists identify and mitigate ethical bias in machine learning models. Machine learning systems can inadvertently perpetuate or amplify societal biases present in training data, leading to unfair outcomes for certain demographic groups.

### Key features:

* Detect bias against unprivileged groups in structured datasets
* Support for preprocessing and postprocessing bias mitigation strategies
* Visualization tools to compare bias metrics before and after mitigation
* Implementation of techniques like Reweighing, Reject Option Classification, and Disparate Impact Remover
* Example notebooks demonstrating bias detection and mitigation on real-world datasets
* Built as a practical wrapper around IBM's AIF360 library, this project simplifies the process of ensuring fairness in ML models when working with sensitive attributes like race, gender, or nationality.

___ 

#### **Why ethical bias mitigation is important?**

Discrimination and bias in machine learning models is an area of hot debate.

* The famous \`*Boston House Prices*\` dataset (available on [Kaggle](https://www.kaggle.com/datasets/vikrishnan/boston-house-prices)) has an ethical problem \- the authors of this dataset engineered a non-invertible variable "B" assuming that racial self-segregation had a positive impact on house prices (Read [this](https://medium.com/@docintangible/racist-data-destruction-113e3eff54a8) article). The purpose of the dataset has since gone from “predicting house prices” to “studying and educating about ethical issues in data science and machine learning”.  
* “There’s software used across the country to predict future criminals. And it’s biased against blacks”. Read this interesting [article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing).

\#TODO: Add a section on types of biases and ethical bias

#### **Ethical Bias mitigation strategies \-** 

Algorithms to detect & mitigate bias and yield fair outcomes fall under 3 categories \- 

1. Preprocessing  
2. Optimising at training time  
3. Postprocessing


### **Python Libraries for Bias & Fairness**

* [AI Fairness 360](https://aif360.mybluemix.net/) \- By IBM ([Git](https://github.com/Trusted-AI/AIF360))  
* [Fair-learn](https://fairlearn.org/) \- By Microsoft ([Git](https://github.com/fairlearn/fairlearn))  
* FairLens ([Git](https://github.com/synthesized-io/fairlens))  
* [What-If-Tool](https://pair-code.github.io/what-if-tool/) \- By Google

#### **AI Fairness 360 (IBM)**

*Pros:*

* Contains bias mitigation algorithms for \- ([link](https://github.com/Trusted-AI/AIF360))  
  * Preprocessing (Eg. Reqeighing)  
  * In-Processing (Eg. Adverserial Debiasing)  
  * Postprocessing (Eg. Thresholding)

*Cons:*

* Multiple privileged groups (sensitive features) are not supported (able to include only a single bias privilege group, therefore the DS will not be able to mitigate model bias in case of multiple bias-inducing features/columns, and will need to check individually. Need to build a wrapper class around AIF360 to tackle this challenge)

#### **Fairlearn (Microsoft)**

* Following Bias Mitigation algorithms are in-built \- [link](https://fairlearn.org/v0.8/user_guide/mitigation.html)  
* Contains algorithms for bias mitigation in both classification and regression-based ML models

*Pros:*

* Can include multiple sensitive features at a time

*Cons:*

* Widget features deprecated in the latest release  
* Processing time for \`mitigating disparity\` is very high  
  * Reductions approach for disparity mitigation is based on Lagrange Multipliers, where disparity constraints are cast as Lagrange multipliers, which causes the reweighting and relabelling of the input data. Since this runs for a number of iterations until convergence is reached, this method is time-consuming, even for the small dataset that we have tested for now (\~20k rows). Need to check how scalable this is for larger datasets.  
* CorrelationRemover \- manual process, cannot be automated

***Other Libraries \-*** largely in an Inactive state or very few stars on their github pages

### **References**

* ***Basics***  
  * [https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias](https://developers.google.com/machine-learning/crash-course/fairness/types-of-bias)  
  * [https://techairesearch.com/most-essential-python-fairness-libraries-every-data-scientist-should-know/](https://techairesearch.com/most-essential-python-fairness-libraries-every-data-scientist-should-know/)  
  * [https://aif360.mybluemix.net/](https://aif360.mybluemix.net/)  
  * [https://aif360.mybluemix.net/resources\#tutorials](https://aif360.mybluemix.net/resources#tutorials)  
  * [https://fairlearn.org/v0.6.2/faq.html](https://fairlearn.org/v0.6.2/faq.html)  
  * [https://www.giskard.ai/knowledge/how-to-test-ml-models-5-the-80-rule-to-measure-disparity\#:\~:text=Disparate%20impact%20in%20AI%20refers,decisions%20that%20discriminate%20against%20them](https://www.giskard.ai/knowledge/how-to-test-ml-models-5-the-80-rule-to-measure-disparity#:~:text=Disparate%20impact%20in%20AI%20refers,decisions%20that%20discriminate%20against%20them).  
  * [https://towardsdatascience.com/tutorial-breaking-myths-about-ai-fairness-the-case-of-biased-automated-recruitment-9ee9b2ecc3a](https://towardsdatascience.com/tutorial-breaking-myths-about-ai-fairness-the-case-of-biased-automated-recruitment-9ee9b2ecc3a)  
  * [https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)  
* ***Research Papers***  
  * [https://www.mdpi.com/2078-2489/13/5/237?type=check\_update\&version=1](https://www.mdpi.com/2078-2489/13/5/237?type=check_update&version=1)

***Current work with AIF360 \-***   
*AIF360 limitations \-*

1. Too many algorithms  
2. Bias diagnosis and mitigation implementation is not straightforward in code (see [example notebooks](https://github.com/Trusted-AI/AIF360/tree/master/examples))  
3. Can only check bias for 1 sensitive feature at a time

Overall code complexity is moderate. To make it easier to work with and make it a “plug-and-play” kind of thing, we can design a wrapper class around AIF360. The idea is \-

![Bias & Fairness Framework](./screenshot-1.png)