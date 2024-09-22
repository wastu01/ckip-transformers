import time
from ckip_transformers import __version__
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker


start_time = time.time()

# Initialize drivers
# 初始化斷詞WS、詞性標記POS、命名實體辨識NER
print("Initializing drivers ... WS")
ws_driver = CkipWordSegmenter(model="bert-base")
print("Initializing drivers ... POS")
pos_driver = CkipPosTagger(model="bert-base")
print("Initializing drivers ... NER")
# ner_driver = CkipNerChunker(model="bert-base")
# print("Initializing drivers ... done")


# 測試文本
sentences = ["我定居於中華民國(台灣)。我來自國立台中教育大學", 
            "今天天氣很好。期許未來的社會會更好。",
            "I am from Taiwan. This is 2024/09/21測試"]


# - 初始化斷詞（WS）、命名實體辨識（NER）輸入格式 ----> 必須是句子的串列（list of sentences）
# - 詞性標記（POS）輸入格式 ----> 必須是詞串列的串列（list of list of words，來自斷詞的輸出）

all_words = []

# 執行斷詞
print("\n斷詞結果：")
ws_results = ws_driver(sentences)
# print(ws_results)
# print(type(ws_results[0]))
for n, sentence in enumerate(ws_results):
    print(f"句子 {n}:")
    print(len(sentence))
    print(f"資料型態: {type(sentence)}")
    # print(type(sentence))
    for word in sentence:
        all_words.append(word)
    # all_words.extend(sentence) 把可迭代對象的元素逐個添加到列表的末尾

print("\n所有詞的列表:")
print(all_words)
print(f"總詞數: {len(all_words)}")

all_pos = []

# 執行詞性標記
print("\n詞性標記結果：")
pos_results = pos_driver(ws_results)
for sentence in pos_results:
    print(sentence)
    print(type(sentence))
    
    for pos in sentence:
        all_pos.append(pos)

print(f"總詞性數: {len(all_pos)}")






# 命名實體識別
print("\n命名實體識別結果：")
ner_results = ner_driver(sentences) # type: ignore
for sentence in ner_results:
    print(sentence)
    print(type(sentence))
# 利用命名實體識別來標記出文本內的專有名詞，接著輸出到辭典裡便可快速建立自定義辭典。

# 結束計時並計算總時間
end_time = time.time()
total_time = end_time - start_time

print(f"\n總執行時間：{total_time:.2f} 秒")



# is_V_or_N = pos_driver(pos)   
 
    
    
# 學習資源：
# https://ithelp.ithome.com.tw/articles/10295882
# https://github.com/ckiplab/ckip-transformers/blob/master/example/example.py
# https://medium.com/@hjeremy1222/簡單好學的中文lda-latent-dirichlet-allocation-主題分類模型-b0a0d2435b60
# 詞性對照表
# https://ckip.iis.sinica.edu.tw/CKIP/paper/poslist.pdf


# sentences = [
#     "我喜歡吃蘋果。",
#     "今天天氣很好。"
# ]

# ws_results = ws_driver(sentences)

# # ws_results 可能看起來像這樣：
# # [
# #     ['我', '喜歡', '吃', '蘋果', '。'],
# #     ['今天', '天氣', '很', '好', '。']
# # ]

# # 因此：
# print(ws_results[0])  # 輸出：['我', '喜歡', '吃', '蘋果', '。']
# print(ws_results[1])  # 輸出：['今天', '天氣', '很', '好', '。']


# # 假設我們有以下斷詞和詞性標記結果
# ws_results = [['我', '喜歡', '吃', '蘋果']]
# pos_results = [['Nh', 'VK', 'VC', 'Na']]

# # 使用 zip 遍歷
# for ws, pos in zip(ws_results[0], pos_results[0]):
#     print(f"詞: {ws}, 詞性: {pos}")


# 文字暫時放置區：
# 涉柯文哲政治獻金案 眾望基金會財報交廉政署偵辦民眾黨主席柯文哲眾望基金會遭疑涉洗錢，且未依法將財報等資料送北市社會局備查。社會局表示，已收到補件並交由廉政署偵辦。廉政...
# 京華廣場「假動土」為拿建照？ 柯文哲「軍令狀」搶救沈慶京五大弊案疑有料服務？-0920【關鍵時刻2200精彩3分鐘】報導本於目前偵辦進度與披露資訊，任何人在依法被判決有罪確定前，均應推定為無罪京華廣場「假動土」為拿建照？ 柯文哲「軍令狀」搶救沈慶京五大弊案疑有料服務？22 分鐘前
# 柯文哲政治獻金「很多匿名捐款」？ 黃珊珊：因為有人懶惰台灣民眾黨主席柯文哲參選年初的總統大選，日前被發現政治獻金申報不實，而其過去在台北市長任內的京華城容積案也遭檢方偵辦，蠟燭多頭燒。民眾黨立委、時任競選總幹事...
# 力挺捍衛柯文哲 蔡壁如：都委會採合議制非市長一人可決定民眾黨主席柯文哲因京華城案遭羈押超過2周，高雄場挺柯宣講活動今晚在鳳山龍成宮登場，前立委蔡壁如在台上表示，她針對北市議員...
# 柯文哲的御三家 黃國昌、蔡壁如、黃珊珊隨著民眾黨主席柯文哲因司法案件被羈押,民眾黨面臨重大挑戰。政治評論家黃暐瀚在暐瀚觀點節目中分析指出, 黃國昌、蔡壁如和黃珊珊已成為民眾黨的"阿北御三家",...
# 聲援柯文哲 民眾黨戶外開講今晚移師高雄民眾黨戶外開講「為司法公義站出來」高雄場今晚6點半在五甲龍成宮廟埕登場，現場擺放500張椅子，開場時仍有不少空位，主辦單位則預估愈晚人潮會愈多，晚間7點多已經坐滿。2 小時前
# 柯文哲弊案延燒沈慶京轉汙點證人機率為零? 鍾小平酸:要錢不要命.上吐下瀉還不招阿北爛尾留給蔣接手?│【驚爆新聞線】20240920│三立新聞台柯文哲弊案延燒沈慶京轉汙點證人機率為零? 鍾小平酸:要錢不要命.上吐下瀉還不招阿北爛尾留給蔣接手?│【驚爆新聞線】20240920│三立新聞台.
# 綜合整理／柯文哲罪狀加一？北市科調查報告出爐，痛批柯市府「為標而標」！北檢動員「開查」約談多名北市府公務員台北地檢署也接獲民眾告發，收案後分他字案偵辦，交由檢肅黑金專組檢察官指揮廉政署蒐證、調卷，案由是「貪污治罪條例」，並將柯文哲列為該案被告⋯⋯3 小時前
# 柯文哲配合沈慶京演戲？ 洪健益質疑京華城「假開工」－民視新聞【民視即時新聞】京華城案繼續燒，台北市議員不分藍綠，緊盯案情，綠營議員"洪健益"再爆料，京華城疑似"假開工"，真正的開工日，不是當初風光的動土典禮2022年10月19日，...
# 柯文哲「1500沈慶京」是指比特幣？陳智菡嗆三立垃圾桶： 鍾小平餵什麼就吃民眾黨主席柯文哲因京華城容積獎勵爭議案涉圖利被羈押禁見


