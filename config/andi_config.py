train_stage = 'FP'
# ======================= #
# QN 训练参数
# ======================= #
qn_cycle = 3

qn_weight_bit_range = [8, 4]
qn_feature_bit_range = [8, 8]
qn_noise_range = [0.01, 0.08]

# ======================= #
# QN 训练参数 - AnDi
# ======================= #
qna_cycle = 2

qna_weight_bit_range = [4, 4]
qna_feature_bit_range = [8, 8]
qna_noise_range = [0.08, 0.1]

# ======================= #
# ADDA 训练参数
# ======================= #
adda_cycle = 3

adda_weight_bit_range = [4, 4]
adda_feature_bit_range = [8, 8]
adda_noise_range = [0.05, 0.08]
adda_adc_bit_range = [8, 8]
adda_dac_bit_range = [5, 5]
