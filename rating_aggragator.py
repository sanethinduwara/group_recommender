
class RatingAggragator:

    @staticmethod
    def average(df_arr):
        _sorted = RatingAggragator.sort_movies_by_id(df_arr)
        _final_rating = _sorted[0].copy()
        _sum = _sorted[0]['rating']
        for x in range(1, len(_sorted)):
            _sum = _sum.add(_sorted[x]['rating'])
        _final_rating.rating = _sum.divide(len(df_arr), 0)
        return _final_rating.sort_values(by=['rating'], ascending=False)

    @staticmethod
    def least_miseary():
        print('least_miseary')

    def sort_movies_by_id(df):
        _sorted_df = []
        for d in df:
            _temp = d.sort_values(by=['movieId'])
            _temp = _temp.set_index('movieId')
            _sorted_df.append(_temp)
        return _sorted_df
