import numpy

from nlp4kor_pytorch.word2vec.word2vec_embedding import Word2VecEmbedding

METRIC = 'cosine'


def test_similar(embedding: Word2VecEmbedding, top_n=3, metric=METRIC):
    """
    find similar words
    :param embedding:
    :param top_n: number of data, default: 3
    :param metric: 'cosine' or 'euclidean', defulat: 'cosine'
    :return:
    """
    for word in ['서울시', '아이폰', '컴퓨터', '왕', '스포츠', '김대중', 'KBS', '사랑', '시작했다.', '한국의']:
        print(f'{word} (freq: {embedding.freq(word):.4f}) -> {embedding.most_similar([word], top_n=top_n, metric=metric)}')


def test_doesnt_match(embedding: Word2VecEmbedding, metric=METRIC):
    for words in [
        ('한국', '미국', '중국', '서울'),
        ('초등학교', '중학교', '고등학교', '학원'),
        ('김영삼', '노태우', '김대중', '노무현', '이건희'),
        ('아이폰', '아이패드', '안드로이드', '맥북'),
        ('삼성전자', 'LG전자', '애플', '네이버'),
        ('기독교', '천주교', '불교', '학교'),
        ('코카콜라', '펩시', '포도주'),
        ('프린터', '마우스', '키보드', '모니터', '선풍기'),
        ('냉장고', '선풍기', '세탁기', '컴퓨터'),
    ]:
        print(words, '->', embedding.doesnt_match(words, top_n=1, metric=metric))


def test_relation(embedding: Word2VecEmbedding, top_n=1, metric=METRIC):
    """
    semantic & syntatic analogy task
    :param embedding:
    :param top_n: number of data, default: 3
    :param metric: 'cosine' or 'euclidean', defulat: 'cosine'
    :return:
    """
    for word1, word2, word3 in [
        ('한국', '서울', '일본'),
        ('왕', '여왕', '남자'),
        ('연구', '연구를', '공부'),
        ('태어난', '태어났다.', '발매된'),
        ('태어난', '태어났다.', '공개된'),
        ('서울', '서울시', '부산'),
        # ('서울', '서울시', '광주'),

        # ('한국', '서울', '중국'),
        # ('한국', '서울', '미국'),  # 미국의 수도가 아니라 영국의 수도가 나옴.
        # ('미국', '워싱턴', '한국'),
        # ('미국', '뉴욕', '한국'),
        # ('시작', '시작했다.', '공부'),
        # ('시작', '시작했다.', '사과'),  # 동음이의어 오류

        ('유치원', '초등학교', '중학교'),
        # ('중학교', '초등학교', '고등학교'),

        ('공개', '공개했다.', '발표'),
        ('공개', '공개하고', '발표'),
        # ('공개', '공개하여', '발표'),

        ('연구', '연구를', '축구'),
        ('농구', '농구를', '축구'),  # 연구-연구를 보다 잘 나옴.
    ]:
        y = embedding.relation_ab2xy(word1, word2, word3, top_n=top_n, metric=metric)
        print(f"{word1} vs {word2} = {word3} vs {y}")


# noinspection PyDefaultArgument


def test_suffix(embedding: Word2VecEmbedding, top_n=3, metric=METRIC):
    for root in ['출시', '공부', '연구', '시작', '종료', '사랑', '살인', '소통']:
        for suffix in ['했다.', '하고', '하여', '했지만', '하는', '하자']:
            # print(root, suffix)
            y_list = embedding.add_suffix(root, suffix, top_n=top_n, metric=metric)
            if len(y_list) > 0:
                print(root, f'+ {suffix} ->', y_list)


def test_root(embedding: Word2VecEmbedding, top_n=1, metric=METRIC):
    text = """서태지가 직접 활동한 것은 전무하며 음반 발매 이후 삼성전자에서 서태지 기념앨범 음원 14곡과 뮤직비디오, 미공개 동영상 등 스페셜 에디션에 내장된 MP3 ‘옙 P2 서태지 스페셜 에디션’을 판매했다.
이 사고의 여파로 첫 프로리그 경기인 삼성전자 칸과 KTF 매직엔스의 대결은 무관중으로 경기를 해야 했다.
하지만 2011년 피처 폰은 삼성전자의 천지인으로 통일하였다.
삼성전자 선수였던 이왕돈과의 사이에 둔 1남 1녀인 이광재(현 부산 KT 소닉붐)와 이유진(현 부천 하나외환이 모두 선수로 뛰고 있는 농구 가족이다.
2009년 상반기 드래프트에서 삼성전자 칸의 2차 지명으로 입단하였다.
삼성물산 경공업본부 뱅커스트러스트 인터내셔널 동경지점 국제자본시장부 부사장 삼성 회장 비서실 국제금융담당 이사 삼성전자 자금팀장 삼성생명 전무 삼성투자신탁운용 대표이사 사장 삼성증권 사장 우리금융 회장 겸 우리은행장 법무법인 세종 고문 KB금융지주 회장 차병원그룹 부회장 차바이오앤디오스텍 대표이사 회장 한국금융투자협회 회장 첫 사회생활은 삼성물산에서 시작했다.
이후 삼성전자와 삼성생명, 삼성투신, 삼성증권 등 다양한 계열사를 거치며 금융과 실물경제 모두를 고루 섭렵했다.
황의 법칙()은 한국의 삼성전자의 기술총괄 사장이었던 황창규가 제시한 이론이다.
현재 김앤장 법률사무소의 상임고문이며, 삼성전자 이사회의 사외이사이다.
에이스침대 1985년 삼성전자 삼성 세탁기 "점보 크리스탈" (Feat. 김민자) 1986년 ~ 1990년 동서식품 동서 차 시리즈 1986년 삼성전자 삼성 세탁기 "센서 크리스탈" (Feat. 김민자, 조용원, 이순재)
2013년 ~ 현재 : KT 대표이사 회장 2013년 ~ 현재 : 부산 KT 소닉붐 구단주 2013년 ~ 현재 : KT 위즈 구단주 2010년 지식경제R&D 전략기획단 단장 2008년 삼성전자 기술총괄 사장 2004년 ~ 2008년 삼성전자 반도체총괄 겸 메모리사업부 사장 1994년 삼성전자 반도체연구소 상무 1987년 미국 인텔사 자문 1985년 미국 스탠퍼드대 전기공학과 책임연구원
2006년 대한민국 최고과학기술인상 2005년 홍콩 아시아머니 선정 아시아 최고경영자 1994년 삼성전자 특별개발포상
2010년 삼성전자 YEPP CF 모델 (With 장근석) 2012년 KBS2 《드림하이 2》 - 나나 역 2014년 Mnet 《No Mercy》 2015년 MBC 《나는 가수다 3》
훈민정음은 삼성전자에서 개발한 윈도용 한글 워드 프로세서로, 자사 컴퓨터에 번들 소프트웨어로 훈민정음을 제공하여 사용자들을 얻었다.
2014년 3분기 기준으로 세계에서 휴대 전화 시장 점유율이 가장 높은 업체는 삼성전자이며 24.7%의 점유율로 작년보단 많이 떨어졌다.
2012년 1분기 기준으로 삼성전자가 노키아를 제치고 세계 휴대 전화 시장 점유율 1위에 올랐다.
미국 시장조사업체 스트래티지어낼리틱스(SA)는 26일(현지시간) “삼성전자가 1분기에 휴대 전화 9350만 대(점유율 25%)를 판매해 14년간 1위를 지킨 노키아를 눌렀다”고 발표했다.
하지만 2011년 피처 폰은 삼성전자의 천지인으로 통일하였다.
"""

    for sentence in text.split('\n'):
        sentence = sentence.strip()
        if len(sentence) == 0:
            continue
        print()
        print(sentence)
        for word in sentence.split(' '):
            _roots = embedding.roots(word, top_n=top_n, metric=metric)
            if len(_roots) > 0:
                print(word, '->', _roots)


def test_sentiment(embedding, metric=METRIC):  # TODO:
    text = """
삼성전자는 실적과 시장점유율 면에서 큰 성장을 이루었다. 1993년 29조원이었던 그룹 매출은 2013년 380조원으로 늘었으며, D램 하나 뿐이던 시장점유율 1위 제품은 20개로 늘어났다.
이 앨범은 한국에서도 120만장 이상이 팔리며 외국가수의 음반으로서는 최다 판매량 기록을 세웠다.
첫 번째 침공은 군사적 무능함과 폭풍으로 실패하였다.
그러나 타격 성적은 .250 미만으로 하락했고 홈런도 1개 뿐에 그쳤다.
"""

    pos_words = ['좋다.', '좋아지다.', '우수하다.', '뛰어나다.', '강하다.', '쉽다', '향상된다.', '올라간다.', '월등히']
    neg_words = ['나쁘다.', '나빠지다.', '부족하다.', '취약하다.', '약하다.', '어렵다.', '저하된다.', '낮아진다.', '현저히']

    senti_words = embedding.most_similar(pos_words, excloude_input=False, top_n=200, metric=metric)
    senti_words.extend(embedding.most_similar(neg_words, excloude_input=False, top_n=200, metric=metric))

    pos_vec = embedding[0]
    neg_vec = embedding[0]
    for word in senti_words:
        pos_vec = embedding.mean(pos_words)
        neg_vec = embedding.mean(neg_words)

        pos_sim = embedding.similarity_vec(pos_vec, embedding[word], metric=metric)
        neg_sim = embedding.similarity_vec(neg_vec, embedding[word], metric=metric)
        # print(word, pos_sim, neg_sim)
        if pos_sim > 0 and neg_sim > 0 and abs(pos_sim - neg_sim) > 0.05:
            if pos_sim > neg_sim:
                pos_words.append(word)
            elif pos_sim < neg_sim:
                neg_words.append(word)

    print('pos_words:', len(pos_words), pos_words[:30])
    print('neg_words:', len(neg_words), neg_words[:30])

    # print('pos_vec:', pos_vec[:10], numpy.count_nonzero(pos_vec))
    # print('neg_vec:', neg_vec[:10], numpy.count_nonzero(neg_vec))
    for sentence in text.split('\n'):
        sentence = sentence.strip()
        if len(sentence) == 0:
            continue
        print()
        print(sentence)
        words = sentence.split(' ')
        weights = embedding.importances(words, metric=metric)
        # for w, i in zip(words, importances):
        #     print(w, i)

        sentiments = []
        for word, weight in zip(words, weights):
            if weight > 0.01:
                pos_sim = embedding.similarity_vec(embedding[word], pos_vec, metric=metric)
                neg_sim = embedding.similarity_vec(embedding[word], neg_vec, metric=metric)

                if pos_sim > 0 and neg_sim > 0 and abs(pos_sim - neg_sim) > 0.01:
                    # print(word, pos_sim > neg_sim, pos_sim, neg_sim)
                    print(f'{pos_sim - neg_sim:.3f} (weight: {weight:.2f}) {word}')  # , f'{pos_sim:.2f}', f'{neg_sim:.2f}')
                    sentiments.append(weight * numpy.sign(pos_sim - neg_sim))

        print(f'sentiment: {numpy.mean(sentiments):.3f}')


def test_chunking(embedding):  # TODO:
    pass


def test_translation(embedding):  # TODO:
    pass


if __name__ == '__main__':
    embedding_file = Word2VecEmbedding.DEFAULT_FILE
    print(embedding_file)
    print()

    embedding = Word2VecEmbedding.load(embedding_file)
    print()
    test_doesnt_match(embedding)
    print()
    test_relation(embedding)
    # print()
    # test_root(embedding)
