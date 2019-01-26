class RedditNews():

  def __init__(self, json):
    self.title = json['title']
    self.content = json['content']

  def getContent(self):
    return '\n\n'.join(self.content)
    