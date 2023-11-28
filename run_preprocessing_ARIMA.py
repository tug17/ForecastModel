from ForecastModel.utils.preprocessing import CreateIndices


DATA_PATH   = r"/home/sebastian/working_dir/Dissertation/LSTM_correction/data_preparation/train_data/Edelsdorf.csv"
OUTPUT_PATH = r"/home/sebastian/working_dir/Dissertation/LSTM_correction/ForecastModel/indices" 

for osc_length in [0, 12, 24]:
    print(f"processing osc_length = {osc_length}")
    ci = CreateIndices(DATA_PATH, out_path=OUTPUT_PATH+f"_{osc_length}")
    ci.create(n_sets = 7, 
            hincast_lengths = [50,75,100,125,150,175,200,225,250,275,300,325,350,375,400,425,450,475,500], 
            forecast_len = 96, 
            target_len = 96, 
            oscilation_len=osc_length)
