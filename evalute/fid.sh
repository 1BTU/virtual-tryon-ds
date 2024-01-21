#评估fid指标
#python -m pytorch_fid ../data/VITON/VITON_test/test_img ../results/TEST_PAIR/original_model # 7.729897758025942
#python -m pytorch_fid ../data/VITON/VITON_test/test_img ../results/TEST_PAIR/pf_b_2l2e-05m_63 # 10.00662242336594
#python -m pytorch_fid ../data/VITON/VITON_test/test_img ../results/TEST_PAIR/pf_b_2l2e-05m_35 #10.221725293830644
#python -m pytorch_fid ../data/VITON/VITON_test/test_img ../results/TEST_PAIR/pf_b_2l2e-05_0 #10.142539174986979
#python -m pytorch_fid ../data/VITON/VITON_test/test_img ../results/TEST_PAIR/pf_b_2l5e-05_0 #9.558380632409637
#评估ssim指标
#python evalute_pic.py -e "ssim" -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/original_model1" #0.8397776847529342
#python evalute_pic.py -e "ssim"  -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/pf_b_2l2e-05m_63" #0.8255207131205364
#python evalute_pic.py -e "ssim"  -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/pf_b_2l2e-05_0" #0.8237090113113051
#python evalute_pic.py -e "ssim"  -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/pf_b_2l5e-05_0" #0.8409505433075558
#python evalute_pic.py -e "ssim"  -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/pf_b_2l2e-05m_35" #0.8427054582532743
#评估psnr指标
#python evalute_pic.py -e "psnr" -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/original_model1" #23.48726880244418
#python evalute_pic.py -e "psnr"  -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/pf_b_2l2e-05m_63" #22.324083498404047
#python evalute_pic.py -e "psnr"  -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/pf_b_2l2e-05_0" #22.217543741316334
#python evalute_pic.py -e "psnr"  -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/pf_b_2l5e-05_0" #22.701533597125096
#python evalute_pic.py -e "psnr"  -p "../data/VITON/test_any.txt" -t "../data/VITON/VITON_test/test_img" -r "../results/TEST_PAIR/pf_b_2l2e-05m_35" #22.88174267229829

#评估IS指标
python evalute_pic.py -e "is"  -r "../results/TEST_PAIR/original_model1" #2.755869947298343
python evalute_pic.py -e "is"  -r "../results/TEST_PAIR/pf_b_2l2e-05m_63" #2.7159903333840436
python evalute_pic.py -e "is"  -r "../results/TEST_PAIR/pf_b_2l2e-05_0" #2.7244085368174717
python evalute_pic.py -e "is"  -r "../results/TEST_PAIR/pf_b_2l5e-05_0" #2.7206951409043234
python evalute_pic.py -e "is"  -r "../results/TEST_PAIR/pf_b_2l2e-05m_35" #2.738346250467623
