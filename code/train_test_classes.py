class_list = ['100_espresso', '101_alp', '102_cliff', '103_reef', '104_lakeside', '105_seashore', '106_salamander',
              '107_acorn', '108_stork', '109_penguin', '10_mask', '110_albatross', '111_dugong', '112_submarine',
              '113_platypus', '114_durian', '115_rickshaw', '116_hovercraft', '117_hedgehog', '118_terrier',
              '119_freighter', '11_go-kart', '120_parrot', '121_tram', '122_turtle', '123_excavator', '124_carp',
              '125_walnut', '126_hazelnut', '127_mangosteen', '128_boat', '129_retriever', '12_gondola',
              '130_shepherd',
              '131_bullfrog', '132_poodle', '133_tabby', '134_cat', '135_cougar', '136_lion', '137_bear',
              '138_ladybug',
              '139_fly', '13_centipede', '140_bee', '141_frog', '142_grasshopper', '143_stick', '144_cockroach',
              '145_mantis', '146_dragonfly', '147_monarch', '148_butterfly', '149_cucumber', '14_hourglass',
              '150_pig',
              '151_hog', '152_alligator', '153_ox', '154_bison', '155_bighorn', '156_gazelle', '157_camel',
              '158_orangutan', '159_chimpanzee', '15_ipod', '160_baboon', '161_elephant', '162_panda',
              '163_constrictor', '164_abacus', '165_gown', '166_altar', '167_apron', '168_backpack',
              '169_bannister',
              '16_kimono', '170_barbershop', '171_barn', '172_barrel', '173_basketball', '174_trilobite',
              '175_bathtub',
              '176_wagon', '177_beacon', '178_beaker', '179_bikini', '17_lampshade', '180_binoculars',
              '181_birdhouse',
              '182_bowtie', '183_brass', '184_scorpion', '185_broom', '186_bucket', '187_train', '188_shop',
              '189_candle', '18_mower', '190_cannon', '191_cardigan', '192_machine', '193_player', '194_chain',
              '195_widow', '196_chest', '197_stocking', '198_dwelling', '199_keyboard', '19_lifeboat', '1_goldfish',
              '200_confectionery', '201_convertible', '202_crane', '203_dam', '204_desk', '205_table',
              '20_limousine',
              '21_compass', '22_maypole', '23_goose', '24_uniform', '25_miniskirt', '26_van', '27_nail', '28_brace',
              '29_obelisk', '2_tarantula', '30_oboe', '31_organ', '32_meter', '33_phone', '34_koala', '35_fence',
              '36_bottle', '37_plunger', '38_pole', '39_poncho', '3_drumstick', '40_wheel', '41_projectile',
              '42_bag',
              '43_jellyfish', '44_reel', '45_refrigerator', '46_remote-control', '47_chair', '48_ball', '49_sandal',
              '4_dumbbell', '50_bus', '51_scoreboard', '52_snorkel', '53_coral', '54_sock', '55_sombrero',
              '56_heater',
              '57_web', '58_sportscar', '59_stopwatch', '5_flagpole', '60_sunglasses', '61_bridge', '62_trunks',
              '63_snail', '64_syringe', '65_teapot', '66_teddy', '67_thatch', '68_torch', '69_tractor',
              '6_fountain',
              '70_arch', '71_trolleybus', '72_turnstile', '73_umbrella', '74_vestment', '75_viaduct',
              '76_volleyball',
              '77_jug', '78_tower', '79_wok', '7_car', '80_spoon', '81_book', '82_plate', '83_guacamole', '84_slug',
              '85_icecream', '86_lolly', '87_pretzel', '88_potato', '89_cauliflower', '8_pan', '90_pepper',
              '91_mushroom', '92_orange', '93_lemon', '94_banana', '95_lobster', '96_pomegranate', '97_loaf',
              '98_pizza', '99_potpie', '9_coat']

animals = ['salamander', 'stork', 'penguin', 'albatross', 'platypus', 'hedgehog', 'parrot', 'turtle', 'carp',
           'retriever', 'gondola', 'shepherd', 'bullfrog', 'poodle', 'tabby', 'cat', 'cougar', 'lion', 'bear',
           'ladybug', 'fly', 'centipede', 'bee', 'frog', 'grasshopper', 'stick', 'cockroach', 'mantis', 'dragonfly',
           'monarch', 'butterfly', 'pig', 'hog', 'alligator', 'ox', 'bison', 'bighorn', 'gazelle', 'camel', 'orangutan',
           'chimpanzee', 'baboon', 'elephant', 'panda', 'constrictor', 'trilobite', 'scorpion', 'goldfish', 'goose',
           'tarantula', 'koala', 'jellyfish', 'snail', 'lobster']


def sub_digit(string):
    for i in range(10):
        string = string.replace(str(i), '')
    string = string.replace('_', '')
    return string


def get_train_test_classes_195_10():
    train_list = class_list[:195]
    test_list = class_list[195:]
    for idx, c in enumerate(train_list):
        train_list[idx] = sub_digit(train_list[idx])
    for idx, c in enumerate(test_list):
        test_list[idx] = sub_digit(test_list[idx])
    return train_list, test_list


def get_train_test_classes_164_41():
    train_list = class_list[:164]
    test_list = class_list[164:]
    for idx, c in enumerate(train_list):
        train_list[idx] = sub_digit(train_list[idx])
    for idx, c in enumerate(test_list):
        test_list[idx] = sub_digit(test_list[idx])
    return train_list, test_list


def get_train_test_animal_classes():
    train_list = animals[:40]
    test_list = animals[40:50]
    for idx, c in enumerate(class_list):
        class_list[idx] = sub_digit(class_list[idx])
    return train_list, test_list


if __name__ == '__main__':
    train, test = get_train_test_classes_195_10()
    print(train)
    print(test)
    print(len(animals))
