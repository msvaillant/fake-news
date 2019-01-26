from RedditNews import RedditNews

class RedditRepository:

  def __init__(self, db):
    self.db = db.fake_news

  def insert(self, news):
    collection = self.db.news

    news_data = {
      'title' : news.title,
      'content' : news.content
    }

    collection.insert_one(news_data)

  def get(self, query={}):
    news_db = []
    collection = self.db.news

    # for id_db in collection.find(query):
    #   news_db.append(self.db.news.find_one({'_id' : id_db['_id']}))

    return collection.find(query)








