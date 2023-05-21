# Chat-Assist 👥🤖

This project is a Flask-based chatbot that uses natural language processing techniques for automated responses. 
It utilizes TF-IDF vectorization and cosine similarity to compute the similarity between user input and preprocessed training data. 
The chatbot provides responses based on the most similar questions in the training data. 💬🔍

<center>

[![forthebadge](https://forthebadge.com/images/badges/built-by-developers.svg)](https://forthebadge.com) &nbsp;
[![forthebadge](https://forthebadge.com/images/badges/open-source.svg)](https://forthebadge.com) &nbsp;
[![forthebadge](https://forthebadge.com/images/badges/check-it-out.svg)](https://forthebadge.com) &nbsp;

</center>

## Prerequisites 📋

- Python 3.x 🐍
- Flask 🌐
- scikit-learn 🧠
- NLTK 📚

## Installation ⚙️

1. Clone the repository:
```
git clone https://github.com/amanhex/Chat-Assist.git
```
2. Navigate to the project directory:
```
cd chatbot-project
```
3. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage 🚀

1. Prepare the training data by updating the **data.txt** file with question-answer pairs.
2. Run the Flask server:
```
python app.py
```
3. Open your web browser and go to **http://localhost:5000** to access the chatbot interface.
4. Enter your query and receive automated responses from the chatbot.

## Customization 🎨

- To customize the chatbot's behavior, you can modify the training data in the **data.txt** file.
- Additionally, you can adjust the similarity threshold in the **get_chatbot_response()** function in **app.py** to control the response accuracy.

## Contributing 🤝

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request.

## License 📄

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/) <br />
This project is licensed under the MIT License.

## Author

- **Amanhex** - [https://github.com/amanhex](https://github.com/amanhex)
