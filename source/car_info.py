brand_dict = {
    "audi" : ["audi"],
    "bmw" : ["bmw"],
    "chevrolet" : ["chevrolet"],
    "dodge" : ["dodge"],
    "ford" : ["ford"],
    "gm" : ["gm"],
    "honda" : ["honda"],
    "hyundai" : ["hyundai"],
    "jeep" : ["jeep"],
    "kia" : ["kia"],
    "mercedes": ["mercedes", "mercedes-benz", "mercedes benz"],
    "mini" : [" mini "],
    "nissan" : ["nissan"],
    "renault" : ["renault"],
    "tesla" : ["tesla"],
    "toyota" : ["toyota"],
    "volkswagen" : ["volkswagen", "vw"]
}

#----------------------------------------------------------------------
#Toyota
#----------------------------------------------------------------------
#It is important that yaris cross comes before yaris, otherwise yaris cross models would be classified as yaris
toyota_models = {
    "yaris cross" : ["yaris cross"],
    "yaris"     : ["yaris"],
    "corolla cross" : ["corolla cross"],
    "corolla"   : ["corolla"],
    "rav4"      : ["rav4"],
    "tacoma"    : ["tacoma"],
    "highlander": ["highlander"],
    "prius prime" : ["prius prime"],
    "prius"     : ["prius"]
}

toyota_corolla_subs = {
    "le": [" le ", "corolla le"],
    "se": [" se ", "corolla se"],
    "ce": [" ce ", "corolla ce"]
}

toyota_corolla_cross_subs = {

}

toyota_highlander_subs = {
}
    
toyota_prius_subs = {
    "xle" : ["xle"],
    "limited" : ["limited"]
}

toyota_rav4_subs = {
    "le" : [" le "],
    "limited" : ["limited"],
    "xle" : ["xle"],
}

toyota_tacoma_subs = {
}

toyota_yaris_cross_subs = {

}

toyota_yaris_subs = {
    "hatchback" : ["hatchback"],
    "sedan" : ["sedan"],
}

toyota_models_to_subs = {
    "corolla_cross" : toyota_corolla_cross_subs,
    "corolla" : toyota_corolla_subs,
    "highlander" : toyota_highlander_subs,
    "prius" : toyota_prius_subs,
    "rav4" : toyota_rav4_subs,
    "tacoma" : toyota_tacoma_subs,
    "yaris cross" : toyota_yaris_cross_subs,
    "yaris" : toyota_yaris_subs
     
}

#----------------------------------------------------------------------
#Kia
#----------------------------------------------------------------------

kia_models = {
    "soul" : ["soul",],
    "sportage" : ["sportage"],
}

kia_soul_subs = {

}
    
kia_sportage_subs = {
    
}

kia_models_to_subs = {
    "soul" : kia_soul_subs,
    "sportage" : kia_sportage_subs,    
}

#----------------------------------------------------------------------
#Ford
#----------------------------------------------------------------------

ford_models = {
    "F150" : ["F 150", "F150", "f150", "f 150", "150"],
    "focus" : ["focus"],
    "bronco" : ["bronco"],
}

ford_models_to_subs = {

}

#----------------------------------------------------------------------
#Honda
#----------------------------------------------------------------------

honda_models = {
    "civic" : ["civic"],
    "odyssey" : ["odyssey"],
    "cr-v" : ["cr-v"],
}

honda_models_to_subs = {

}

#----------------------------------------------------------------------
#Nissan
#----------------------------------------------------------------------

nissan_models = {
    "rogue" : ["rogue"],
}

nissan_models_to_subs = {

}
    
#----------------------------------------------------------------------
#Volkswagen
#----------------------------------------------------------------------

volkswagen_models = {
    "jetta" : ["jetta"],
    "tiguan" : ["tiguan"],
}

volkswagen_jetta_subs = {
    "trendline" : ["trendline"],
    "comfortline" : ["comfortline"],
    "highline" : ["highline"]
}

volkswagen_models_to_subs = {
    "jetta" : volkswagen_jetta_subs,
}

#----------------------------------------------------------------------
#Mini
#----------------------------------------------------------------------

mini_models = {
    "john cooper works" : ["john cooper works", "jcw"],
    "countryman" : ["countryman"],
    "cooper" : ["cooper"],
    "cooper s": ["cooper s"],
}

mini_jcw_subs = {
    "countryman" : ["countryman"],
    "convertible" : ["convertible"],
}

mini_countryman_subs = {
    "cooper" : ["cooper"],
    "cooper s" : ["cooper s"]
}

mini_cooper_subs = {
    "3 door" : ["3 door", "3 porte"],
    "5 door" : ["5 door", "5 porte"],
    "convertible" : ["convertible"]
}

mini_models_to_subs = {
    "john cooper works" : mini_jcw_subs,
    "countryman" : mini_countryman_subs,
    "cooper" : mini_cooper_subs
}

#-----------------------------------------------------------------------

brand_to_models = {
    "toyota" : toyota_models,
    "kia" : kia_models,
    "ford" : ford_models,
    "honda" : honda_models,
    "nissan" : nissan_models,
    "volkswagen" : volkswagen_models,
    "mini" : mini_models,
}

brand_models_to_subs = {
    "toyota" : toyota_models_to_subs,
    "kia" : kia_models_to_subs,
    "ford" : ford_models_to_subs,
    "honda" : honda_models_to_subs,
    "nissan" : nissan_models_to_subs,
    "volkswagen" : volkswagen_models_to_subs,
    "mini" : mini_models_to_subs,

}