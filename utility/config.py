OUTPUT_DIR = './output_books/'
RESOURCES_DIR = './resources/'

jpg_options = {
    "quality"    : 100,
    "progressive": True,
    "optimize"   : False
}


model_config = 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config'
extra_config = "MODEL.ROI_HEADS.SCORE_THRESH_TEST"
label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}

storeMaskedImages = True

# Table OCR Related Properties
model_path = './../../models/table_detection_model.pth'
det_threshold = 0.5
table_recognition_language = 'eng'
row_determining_threshold = 0.6667
col_determining_threshold = 0.5
nms_table_threshold = 0.1
nms_cell_threshold = 0.0001


DIRECTORIES = ['Images','CorrectorOutput', 'Comments', 'Inds', 'Dicts', 'ProcessedImages', 'VerifierOutput', 'MaskedImages']


TESSDATA_DIR_CONFIG = '--oem 3 --psm 6'# --tessdata-dir "/home/ayush/udaan-deploy-flask/udaan-deploy-pipeline/tesseract-exec/share/tessdata/"'

IMAGE_CONVERT_OPTION = True
DPI = 300
JPEGOPT={
    "quality"     : 100,
    "progressive" : True,
    "optimize"    : False
}


REC_MODEL_PATH = './models/ocr/handwritten/crnn_vgg16_bn_handwritten_'

VOCABS = {}
VOCABS['bengali'] = 'শ০ূৃৰজআঔঅঊিঢ়খ৵পঢই৳ফঽ৪লেঐযঃঈঠুধড়৲ৄথভটঁঋৱরডৢছ৴ঙওঘস১৹ণগ৷৩ত৮হ৭োষৎ৶কন৬চমৈা়ীৠঝএ৻ব৯য়উৌঞ৺২ংৣদ৫্ৗ-।'
VOCABS['gujarati'] = '૮લઔ૨સખાઑઈઋૐઓવૄ૦઼ઁનઞઊ૫ીશફણ૬૭બ૧રળૌુઠઐઉષપેઇઅૃઝજૉક૱૯ગઍદો૪ૅએંહડઘ૩ૂછઙઃઽટતધિૈયઢ્આમથચભ-'
VOCABS['gurumukhi'] = 'ਗ਼ਵਨਁਰਊਖਂਆਜੈਲੴਣ੧ਛਭਫ੮੯ਚਔੀਯਹਲ਼ਞ੩ੜਫ਼ੁਮ੫ਤੇਦਸ਼ਟੰ੭ਓਅਃਡਾਉਠੱਈ੦ੵਖ਼ਏਕਥ੬ਧੲੑਝਿ੨ਐਬਪਘਸ਼ਙੌਜ਼ੋਗ੍ੳਇ੪ੂਢ-।'
VOCABS['kannada'] = 'ಚೕಒಉೖಂಲಾಝಟೆಅ೬ೇ೨ಬಡವಜಢಞಔಏಧಶಭತಳೀಕಐಈಠಪ೫ಣ೮ೞಆಯುಗೢಋದಘೂ್ೈ೦ಓಱಃಹ೯ೋಮ೭ೠಥಖಫಇರ೪ಛಙೣಿ೩ೌೄಷಌಸನ಼ಊಎ೧ೃೊ-'
# VOCABS['malayalam'] = '൪ഉ൮ള൵ഔംസഞഎഷ൫ൄൌ-ഃൈീഌഛഇണാഈഹധ൭ജച൱൴൹യതൻശഒ൯ഗർഊആവഖൠൣ൩ോൽ൧അ൳ൗപഭൃ്മെഐൡഓദഏറിഠരൺ൰ൾങട൦ഢൢഡലേഴഝൊ൲ബനൂഥൿഘഫുഋക൬൨ '
VOCABS['malayalam'] = ' -ംഃഅആഇഈഉഊഋഌഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരറലളഴവശഷസഹാിീുൂൃൄെേൈൊോൌ്ൗൠൡൢൣ൦൧൨൩൪൫൬൭൮൯൰൱൲൳൴൵൹ൺൻർൽൾൿ'
VOCABS['odia'] = 'ଖ୯୬ୋଓଞ୍ଶ୪ଣଥଚରୄତଃେ୮ଆକଵୂନଦ୰ୖୢଜଉଳଅଁଲଯଔପ୭ଷଢଡ଼ଊୟମିୁ୧ଂ଼ୀବଟଭଢ଼୦ଘଠୗ୫ୡାଐ୨ଙହଈୱ୩ୃଛଏୌଗଫସଇଧଡଝୈୣୠଋ-।'
VOCABS['tamil'] = 'ய௴ஷ௫ைெஸஎஈோவ௲ூு௭அ்ஶி௰ஹ௧ௐா௮ஔ௺சீண௩இனஆழ௪௯ஙஊதஜ௷௶மௌள௸ஐபநேற௬டஒ௹ஞஉஏகௗொர௱௵ஃ௨லஓ௳௦-'
VOCABS['telugu'] = '౦ఱకఆఋడత౯౻ిహౌ౭౽ఉ౮్ధఓగ౼మ౫ూౠఔాఇనైఁజీౄుేసశృఃఝఢరఠలోఞౘఅ౹౧ౢఛబ౸ఐయ౩ఖటచెొఊదఈషథభఏౙ౬౾ఎ౪ణఒప౨ఫంఘఙళవ౺-'
VOCABS['urdu'] = 'ٱيأۃدےش‘زعكئںسحٰنؐةقذ؟ؔ۔—ًمھٗپغٖطإؒرڑصټٍگاؤجضْﷺچ‎ۓِّؓٹظىتڈ‍یُه،خو؛آفبؑلہثﺅ‌ژَۂءک‏'
VOCABS['hindi'] = 'ॲऽऐथफएऎह८॥ॉम९ुँ१ं।षघठर॓ॼड़गछिॱटऩॄऑवल५ढ़य़अञसऔयण॑क़॒ौॽशऍ॰ूीऒॊख़उज़ॻॅ३ओऌळनॠ०ेढङ४़ॢग़पऊॐज२डैभझकआदबऋखॾ॔ोइ्धतफ़ईृःा६चऱऴ७-'
VOCABS['sanskrit']='ज़ऋुड़ऍऐक५टय४उः३ॠध९्७ू१वऌौॐॡॢइ६ाै८नृअंथढेखऔघग़०लजोईरञपफँझभषॅॄगतचहसीढ़आशए।म२दठङबिऊडओळछण़ऽ'
VOCABS['devanagari'] = 'रचख़३ॾऍृेञलॻॉऴषॐॢ१य०ॽएा२ई।ग़७टऐय़॥तोदऽभुनओऒ-ठँ.ौ्८ॼझॠविःक़ी॰छॅॊऩऱ़थजशळङअऋखबफउ५फ़६ऊॲॆज़कढ़मूस॓इऔह॑ैगढॣधआड़९ं४डणपॄघऑ'


MODEL = {
    "eng" : "default", 
    "eng_Latn": "default",
    "asm_Beng": "bengali",
    "hin_Deva": "hindi",
    "mar_Deva": "hindi",
    "tam_Taml": "tamil",
    "ben_Beng": "bengali",
    "kan_Knda": "kannada",
    "ory_Orya": "odia",
    "tel_Telu": "telugu",
    "guj_Gujr": "gujarati",
    "mal_Mlym": "malayalam",
    "pan_Guru": "gurumukhi",
    "snd_Deva": "hindi",
    "urd_Arab": "urdu",
    "gom_Deva": "hindi",
    "brx_Deva": "hindi",
    "devanagari": "hindi",
    "gujarati": "gujarati",
    "gurmikhi": "gurumukhi",
    "gurmukhi": "gurumukhi",
    "bengali": "bengali",
    "kannada": "kannada",
    "malayalam": "malayalam",
    "tamil": "tamil",
    "telugu": "telugu",
    "oriya": "odia",
    "hin": "hindi",
    "kan": "kannada",
    "tam": "tamil",
    "tel": "telugu",
    "ben": "bengali",
    "guj": "gujarati",
    "mal": "malayalam",
    "pan": "gurumukhi",
    "urd": "urdu",
    "asm": "bengali",
    "ori": "odia",
    "gur": "gurumukhi",
    "mar": "hindi",
    "snd": "hindi",
    "gom": "hindi",
    "brx": "hindi",
    "ory": "odia",
    "nep": "hindi",
}


# lang_mapping = {
# "devanagari":"Sanskrit",
# "devanagari": "Nepali",
# "devanagari": "English and Hindi",
# "devanagari":"English and Marathi",
# "devanagari":"English and Nepali",
# "devanagari":"English and Sanskrit",
# "brx_Deva":"Bodo",
# "gom_Deva":"Konkani",
# "urd_Arab": "Urdu",
# "snd_Deva": "Sindhi",
# "gujarati": "English and Gujarati",
# "gurumukhi": "English and Punjabi",
# "bengali": "English and Bengali",
# "kannada":"English and Kannada",
# "malayalam":"English and Malayalam",
# "tamil":"English and Tamil",
# "telugu":"English and Telugu",
# "oriya":"English and Odiya",


# "eng_Latn" : "English",
# "hin_Deva" : "Hindi",
# "guj_Gujr" : "Gujarati",
# "mar_Deva" : "Marathi",
# "pan_Guru" : "Punjabi",
# "ben_Beng" : "Bengali",
# "kan_Knda" : "Kannada",
# "mal_Mlym" : "Malayalam",
# "tam_Taml" : "Tamil",
# "tel_Telu" : "Telugu",
# "asm_Beng" : "Assamese",
# "ory_Orya" : "Odiya"

# }