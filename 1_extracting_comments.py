# Python: using youtube-comment-downloader (if avoiding API limits)
from youtube_comment_downloader import YoutubeCommentDownloader

downloader = YoutubeCommentDownloader()
comments = downloader.get_comments_from_url('https://www.youtube.com/watch?v=OdUzizXtQt8')

# Store comments in a list
comment_list = []
for i in comments:
    comment_list.append(i['text'])  # Or comment['textOriginal']

for i, c in enumerate(comment_list[:10]):
    print(f"{i+1}. {c}")


