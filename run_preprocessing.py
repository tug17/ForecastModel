from ForecastModel.utils.preprocessing import CreateIndices


DATA_PATH   = r"F:\11_EFFORS\data\Edelsdorf.csv"
OUTPUT_PATH = r"F:\11_EFFORS\data\indices" 

for osc_length in [0, 12, 24]:
    print(f"processing osc_length = {osc_length}")
    ci = CreateIndices(DATA_PATH, out_path=OUTPUT_PATH+f"_{osc_length}")
    ci.create(n_sets = 7, 
            hincast_lengths = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120], 
            forecast_len = 96, 
            target_len = 96, 
            oscilation_len=osc_length)
