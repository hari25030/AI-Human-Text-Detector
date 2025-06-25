AI vs Human Text Detector 
my project it tries to guess if text is from a human or AI
What it Does
this app lets you:
	•	upload pdf or word docs
	•	type or paste text
	•	pick an AI model to use (SVM decision tree or adaboost)
	•	get instant guesses human or AI with how sure it is
	•	see charts showing important words
	•	get facts about your text like word count
	•	compare what all 3 models guess on the same text
	•	see how my models did in big tests
	•	download a pdf report of all the info
How to Run My Project
you need python 3 installed
1. Get My Code
first get the project files
git clone https://github.com/hari25030/AI-Human-Text-Detector
cd HariAIclassifier


2. Set Up Python Stuff
make a special folder for python libraries just for this project
python -m venv venv


then turn it on:
	•	Windows: .\venv\Scripts\activate



	•	macOS/Linux: source venv/bin/activate



3. Install Libraries
now install all the python libraries I used they are in requirements.txt
pip install -r requirements.txt


4. Folder Stuff (Important!)
make sure your project folders look like this:
HariAIclassifier
├── app.py                      # app code 
├── requirements.txt            # list of libraries
├── models                     # 3 models 
│   ├── svm_model.joblib
│   ├── decision_tree_model.joblib
│   ├── adaboost_model.joblib
│   └── vectorizer.joblib
├── data                       
│   ├── training_data          #  training excel file
│   └── test_data              #  test csv
├── notebooks                  # my jupyter notebook is here
│   └── AI_vs_Human_Text_Classifierfinal.ipynb
└── README.md                   #  file you are reading


5. Run the App
to start the web app
streamlit run app.py


it opens in your browser
6. See My Jupyter Notebook
to see how I trained everything
jupyter notebook notebooks/AI_vs_Human_Text_Classifierfinal.ipynb


Problems I Had 
	•	'csr_matrix' Error: sometimes a model gave a weird error with numbers I fixed it by making numbers normal first.
	•	Making Explanations: it was hard to explain why AI guessed something I tried to make it sound like a person talking not a robot.
	•	Keeping Code Neat: many files and folders sometimes messy learned to keep it organized.
