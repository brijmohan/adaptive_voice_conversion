. etape.config

if [ $stage -le 0 ]; then
    python make_datasets_etape.py $raw_data_dir/wav48 $raw_data_dir/french_etape_spk_dur.txt $data_dir $n_out_speakers $test_prop $sample_rate $n_utt_attr
fi

if [ $stage -le 1 ]; then
    python reduce_dataset.py $data_dir/train.pkl $data_dir/train_$segment_size.pkl $segment_size
fi

if [ $stage -le 2 ]; then
    # sample training samples
    python sample_single_segments.py $data_dir/train.pkl $data_dir/train_samples_$segment_size.json $training_samples $segment_size
fi
if [ $stage -le 3 ]; then
    # sample testing samples
    python sample_single_segments.py $data_dir/in_test.pkl $data_dir/in_test_samples_$segment_size.json $testing_samples $segment_size
    python sample_single_segments.py $data_dir/out_test.pkl $data_dir/out_test_samples_$segment_size.json $testing_samples $segment_size
fi
