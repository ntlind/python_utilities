

def test_update_time():
    def get_large_test_example():
        """
        Returns a larger, mocked-up dataframe for testing purposes
        """
        example = pd.DataFrame([
        ['202001', 'Cat_1', 213, 'Prod_3', 'CA', 'Store_1'],
        ['202003', 'Cat_1', 123, 'Prod_4', 'CA', 'Store_1'],
        ['202004', 'Cat_1', 2333, 'Prod_4', 'CA', 'Store_1'],
        ['202001', 'Cat_2', 3, 'Prod_5', 'CA', 'Store_1'],
        ['202051', 'Cat_2', 4, 'Prod_6', 'CA', 'Store_1'],
        ['202052', 'Cat_2', 6, 'Prod_6', 'CA', 'Store_1'],
        ['202001', 'Cat_1', 123, 'Prod_3', 'CA', 'Store_2'],
        ['202003', 'Cat_1', 14, 'Prod_4', 'CA', 'Store_2'],
        ['202004', 'Cat_1', 5512, 'Prod_4', 'CA', 'Store_2'],
        ['202001', 'Cat_2', 421, 'Prod_6', 'CA', 'Store_2'],
        ['202051', 'Cat_2', 234, 'Prod_6', 'CA', 'Store_2'],
        ['202052', 'Cat_2', 512, 'Prod_6', 'CA', 'Store_2']
        ], columns=['week', 'category', 'sales', 'product', 'state', 'store'])

        return example

    # get data
    large_df = get_large_test_example()
    large_df['week'] = large_df['week'].astype(int)
    large_df['datetime'] = pd.to_datetime(large_df[TIME_VAR].astype(str) + '-1', format="%Y%U-%w")

    # week tests
    test = update_time(large_df['week'], 53)
    answers = pd.Series(['202102', "202104", "202105", "202102", "202152"]).astype(int)
    assert np.array_equal(test.head(5), answers), print(test.head(5), answers)

    test = update_time(large_df['week'], -53)
    answers = pd.Series(['201852', "201902", "201903", "201852", "201950"]).astype(int)
    assert np.array_equal(test.head(5), answers), print(test)

    test = update_time(large_df['week'], -4)
    answers = pd.Series(['201949', "201951", "201952", "201949", "202047"]).astype(int)
    assert np.array_equal(test.head(5), answers), print(test)

    test = update_time(large_df['week'], -4*7, "days")
    assert np.array_equal(test.head(5), answers), print(test)

    test = update_time(large_df['week'], 5)
    answers = pd.Series(['202006', "202008", "202009", "202006", "202104"]).astype(int)
    assert np.array_equal(test.head(5), answers), print(test)


    # month tests
    test = update_time(large_df['MonthYear'], 5, time_unit='months', datetime_format="%Y%m")
    answers = pd.Series([202006, 202006, 202006, 202006, 202105, 202105])
    assert np.array_equal(test.head(6), answers), print(test)

    test = update_time(large_df['MonthYear'], -12, time_unit='months', datetime_format="%Y%m")
    answers = pd.Series([201901, 201901, 201901, 201901, 201912, 201912])
    assert np.array_equal(test.head(6), answers), print(test)


    # day tests
    test = update_time(large_df['DayYear'], 1, datetime_format="%j")
    answers = pd.Series([13, 27, 34, 13, 363, 5])
    assert np.array_equal(test.head(6), answers), print(test)

    test = update_time(large_df['DayYear'], -6, time_unit='days', datetime_format="%j")
    answers = pd.Series([365, 14, 21, 365, 350, 357])
    assert np.array_equal(test.head(6), answers), print(test)

    test = update_time(large_df['DayYear'], -365, time_unit='days', datetime_format="%j")
    answers = large_df['DayYear'].head(6)
    assert np.array_equal(test.head(6), answers), print(test)


    # int tests
    assert update_time(202005, 5) == 202010
    assert update_time(202005, -10) == 201947
    assert update_time(202005, 53) == 202106
    assert update_time(202005, -53) == 201904

    assert update_time(123, 5, datetime_format="%j", time_unit='days') == 128
    assert update_time(123, 365, datetime_format="%j", time_unit='days') == 123
    assert update_time(123, -370, datetime_format="%j", time_unit='days') == 118

    assert update_time(20201, 5, datetime_format="%Y%m", time_unit='months') == 202006

    assert update_time(234, 5, time_unit='int') == 239
    assert update_time(20201, 19, time_unit='int') == 20220