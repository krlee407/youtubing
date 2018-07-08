import sqlite3

DB_DIR = 'data/youtubing.db'

con = sqlite3.connect(DB_DIR)
cursor = con.cursor()

cursor.execute("CREATE TABLE video_meta \
                (video_id INTEGER PRIMARY KEY AUTOINCREMENT, \
                 title TEXT, \
                 uploaded_date TEXT, \
                 summary TEXT, \
                 url TEXT, \
                 keyword TEXT, \
                 hit_count INTEGER, \
                 like_count INTEGER, \
                 unlike_count INTEGER, \
                 subscribe_count INTEGER, \
                 created_date TEXT)"
)

cursor.execute("CREATE TABLE subtitle_meta \
                (subtitle_id INTEGER PRIMARY KEY AUTOINCREMENT, \
                 filename TEXT, \
                 language TEXT, \
                 is_auto_generated BOOLEAN, \
                 video_id INTEGER, \
                 FOREIGN KEY(video_id) REFERENCES video_meta(video_id))"
)

cursor.execute("CREATE TABLE subtitle_token \
                (subtitle_token_id INTEGER PRIMARY KEY AUTOINCREMENT, \
                 start_time TEXT, \
                 end_time TEXT, \
                 subtitle_token TEXT, \
                 subtitle_id INTEGER, \
                 FOREIGN KEY(subtitle_id) REFERENCES subtitle_meta(subtitle_id))"
)

cursor.execute("CREATE TABLE sentence_meta \
								(sentence_id INTEGER PRIMARY KEY AUTOINCREMENT, \
								 start_time TEXT, \
								 end_time TEXT, \
								 sentence TEXT, \
								 text_token TEXT, \
								 embedding_vector TEXT, \
								 subtitle_id INTEGER, \
								 FOREIGN KEY(subtitle_id) REFERENCES subtitle_meta(subtitle_id))"
)

cursor.close()