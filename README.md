# fake-news
R'n'D project for fake news detection

## List of Files :
- RedditScrapper: Class made to get a number of articles from multiple subredits.
  It fill a folder with the numbered news in extension .news .
- emotion_words: JSON file that contain a list of words used to convince
- Text2Vect: Python Script that transform a text into a vector, it's used with ./Text2Vect.py "WORDFILE" "TEXT"
- ResultVector: File that contain every vector result of Text2Vect
- generateVectors : Shell script that apply Text2Vect to every Article in news/
