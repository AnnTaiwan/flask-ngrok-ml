import ssl
from pytubefix import YouTube
from pytubefix.cli import on_progress

ssl._create_default_https_context = ssl._create_stdlib_context


if __name__ == "__main__":    
    yt = YouTube('https://www.youtube.com/watch?v=R5teHn6pmLs', on_progress_callback=on_progress)   
    # print(yt.streams.all()) # to see the suitable quality of video
    # print(yt.streams.filter(progressive=True))
    print("title:", yt.title)           # 影片標題
    print("length: ", yt.length)          # 影片長度 ( 秒 )
    print("author: ", yt.author)          # 影片作者
    print("channel_url: ", yt.channel_url)     # 影片作者頻道網址
    print("thumbnail_url: ",yt.thumbnail_url)   # 影片縮圖網址
    print("Number of views: ", yt.views)           # 影片觀看數


    print('download...')
    yt.streams.filter().get_highest_resolution().download(filename='Jodie_Comer_reward_360p.mp4')
    # 下載最高畫質影片，如果沒有設定 filename，則以原本影片的 title 作為檔名
    print('\nok!')