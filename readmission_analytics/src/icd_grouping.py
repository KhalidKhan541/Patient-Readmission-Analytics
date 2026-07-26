"""
ICD Code Grouping Module
========================
ICD code grouping via CASE hierarchies for clinical categorization.
Maps ICD-9 and ICD-10 codes to chapters, disease categories, and
Elixhauser comorbidity groups for readmission analytics.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ICDGrouper:
    """Group ICD codes into clinical categories using CASE hierarchies.

    Supports both ICD-9 (numeric, 3-5 digits with optional decimal) and
    ICD-10 (letter + 2-4 alphanumeric characters) code formats.  Version
    detection is automatic based on code format.

    Examples
    --------
    >>> grouper = ICDGrouper()
    >>> grouper.get_chapter("I50.9")
    'Diseases of the circulatory system'
    >>> grouper.get_elixhauser_group("E11.9")
    'diabetes'
    """

    # ------------------------------------------------------------------
    # ICD-10 chapter mapping (first character → chapter)
    # ------------------------------------------------------------------

    ICD10_CHAPTERS: dict[str, str] = {
        'A': 'Certain infectious and parasitic diseases',
        'B': 'Certain infectious and parasitic diseases',
        'C': 'Neoplasms',
        'D': 'Diseases of the blood and blood-forming organs',
        'E': 'Endocrine, nutritional and metabolic diseases',
        'F': 'Mental and behavioural disorders',
        'G': 'Diseases of the nervous system',
        'H': 'Diseases of the eye and adnexa / ear',
        'I': 'Diseases of the circulatory system',
        'J': 'Diseases of the respiratory system',
        'K': 'Diseases of the digestive system',
        'L': 'Diseases of the skin and subcutaneous tissue',
        'M': 'Diseases of the musculoskeletal system and connective tissue',
        'N': 'Diseases of the genitourinary system',
        'O': 'Pregnancy, childbirth and the puerperium',
        'P': 'Certain conditions originating in the perinatal period',
        'Q': 'Congenital malformations, deformations and chromosomal abnormalities',
        'R': 'Symptoms, signs and abnormal clinical and laboratory findings',
        'S': 'Injury, poisoning and certain other consequences of external causes',
        'T': 'Injury, poisoning and certain other consequences of external causes',
        'V': 'External causes of morbidity and mortality',
        'W': 'External causes of morbidity and mortality',
        'X': 'External causes of morbidity and mortality',
        'Y': 'External causes of morbidity and mortality',
        'Z': 'Factors influencing health status and contact with health services',
    }

    # ------------------------------------------------------------------
    # ICD-9 chapter mapping (first 1-2 digits → chapter)
    # ------------------------------------------------------------------

    ICD9_CHAPTERS: dict[str, str] = {
        '001': 'Intestinal infectious diseases',
        '002': 'Intestinal infectious diseases',
        '003': 'Intestinal infectious diseases',
        '004': 'Intestinal infectious diseases',
        '005': 'Intestinal infectious diseases',
        '006': 'Intestinal infectious diseases',
        '007': 'Intestinal infectious diseases',
        '008': 'Intestinal infectious diseases',
        '009': 'Intestinal infectious diseases',
        '010': 'Tuberculosis',
        '020': 'Zoonotic bacterial diseases',
        '030': 'Other bacterial diseases',
        '080': 'Louse-borne relapsing fever',
        '081': 'Other relapsing fevers',
        '090': 'Venereal diseases',
        '100': 'Other diseases caused by spirochetes',
        '110': 'Mycoses',
        '120': 'Helminthiases',
        '130': 'Protozoal diseases',
        '140': 'Neoplasms of lip, oral cavity and pharynx',
        '150': 'Neoplasms of digestive organs',
        '160': 'Neoplasms of respiratory and intrathoracic organs',
        '170': 'Neoplasms of bone and connective tissue',
        '174': 'Neoplasms of female breast',
        '175': 'Neoplasms of female breast',
        '179': 'Neoplasms of female genital organs',
        '180': 'Neoplasms of female genital organs',
        '190': 'Neoplasms of eye, brain and other parts of central nervous system',
        '195': 'Neoplasms of other and ill-defined sites',
        '196': 'Neoplasms of other and ill-defined sites',
        '197': 'Secondary malignancies',
        '198': 'Secondary malignancies',
        '199': 'Malignant neoplasm without specification of site',
        '200': 'Neoplasms of lymphoid and hematopoietic tissue',
        '208': 'Neoplasms of lymphoid and hematopoietic tissue',
        '210': 'Benign neoplasms',
        '220': 'Benign neoplasms',
        '229': 'Benign neoplasms',
        '230': 'Ca in situ',
        '239': 'Neoplasms of unspecified nature',
        '240': 'Endocrine diseases',
        '241': 'Endocrine diseases',
        '242': 'Endocrine diseases',
        '243': 'Endocrine diseases',
        '244': 'Endocrine diseases',
        '245': 'Endocrine diseases',
        '246': 'Endocrine diseases',
        '249': 'Endocrine diseases',
        '250': 'Diabetes mellitus',
        '251': 'Endocrine diseases',
        '252': 'Endocrine diseases',
        '253': 'Endocrine diseases',
        '254': 'Endocrine diseases',
        '255': 'Endocrine diseases',
        '256': 'Endocrine diseases',
        '257': 'Endocrine diseases',
        '258': 'Endocrine diseases',
        '259': 'Endocrine diseases',
        '260': 'Nutritional deficiencies',
        '270': 'Metabolic disorders',
        '280': 'Diseases of the blood',
        '290': 'Mental disorders',
        '300': 'Neurotic disorders',
        '301': 'Personality disorders',
        '302': 'Sexual deviations',
        '303': 'Alcohol dependence syndrome',
        '304': 'Drug dependence',
        '305': 'Nondependent abuse of drugs',
        '306': 'Physiological malfunction arising from mental factors',
        '307': 'Special symptoms or syndromes',
        '308': 'Acute reaction to stress',
        '309': 'Adjustment reaction',
        '310': 'Specific nonpsychotic mental disorders following organic brain damage',
        '311': 'Depressive disorder, not elsewhere classified',
        '312': 'Conduct disorders',
        '313': 'Disturbance of conduct not elsewhere classified',
        '314': 'Hyperkinetic syndrome of childhood',
        '315': 'Specific delays in development',
        '316': 'Mental disorders',
        '317': 'Mental retardation',
        '318': 'Mental retardation',
        '319': 'Mental retardation, severity unspecified',
        '320': 'Bacterial infections of the central nervous system',
        '330': 'Diseases of the nervous system',
        '340': 'Demyelinating diseases of the central nervous system',
        '341': 'Demyelinating diseases of the central nervous system',
        '342': 'Cerebral palsy',
        '343': 'Cerebral palsy',
        '344': 'Other paralytic syndromes',
        '345': 'Epilepsy',
        '346': 'Migraine',
        '347': 'Paroxysmal sleep disorders',
        '348': 'Other conditions of brain',
        '349': 'Other conditions of the nervous system',
        '350': 'Trigeminal nerve disorders',
        '351': 'Facial nerve disorders',
        '352': 'Disorders of other cranial nerves',
        '353': 'Nerve root and plexus disorders',
        '354': 'Mononeuritis of upper limb',
        '355': 'Mononeuritis of lower limb',
        '356': 'Hereditary and idiopathic peripheral neuropathy',
        '357': 'Inflammatory and toxic neuropathy',
        '358': 'Neuromuscular junction disorders',
        '359': 'Diseases of myoneural junction and muscle',
        '360': 'Diseases of the eye',
        '370': 'Diseases of the conjunctiva',
        '371': 'Diseases of the cornea',
        '372': 'Diseases of the sclera, episclera, tenon capsule and orbit',
        '373': 'Diseases of the iris and ciliary body',
        '374': 'Diseases of the retina and optic nerve',
        '375': 'Diseases of the lacrimal system',
        '376': 'Diseases of the orbit',
        '377': 'Diseases of the optic nerve and visual pathways',
        '378': 'Strabismus and other disorders of binocular vision',
        '379': 'Other disorders of eye',
        '380': 'Diseases of the ear and mastoid process',
        '381': 'Diseases of the middle ear and mastoid',
        '382': 'Inflammatory disease of the ear',
        '383': 'Diseases of the pinna',
        '384': 'Diseases of the tympanic membrane',
        '385': 'Diseases of the middle ear and mastoid',
        '386': 'Vertiginous syndromes',
        '387': 'Otosclerosis',
        '388': 'Other disorders of ear',
        '389': 'Hearing loss',
        '390': 'Diseases of the circulatory system',
        '400': 'Hypertensive disease',
        '401': 'Essential hypertension',
        '402': 'Hypertensive heart disease',
        '403': 'Hypertensive renal disease',
        '404': 'Hypertensive heart and renal disease',
        '405': 'Secondary hypertension',
        '410': 'Acute myocardial infarction',
        '411': 'Other acute and subacute forms of ischemic heart disease',
        '412': 'Old myocardial infarction',
        '413': 'Angina pectoris',
        '414': 'Other forms of chronic ischemic heart disease',
        '415': 'Pulmonary heart disease',
        '416': 'Chronic pulmonary heart disease',
        '417': 'Other diseases of pulmonary vessels',
        '420': 'Acute pericarditis',
        '421': 'Acute and subacute endocarditis',
        '422': 'Acute myocarditis',
        '423': 'Other diseases of pericardium',
        '424': 'Other diseases of endocardium',
        '425': 'Cardiomyopathy',
        '426': 'Conduction disorders',
        '427': 'Cardiac dysrhythmias',
        '428': 'Heart failure',
        '429': 'Ill-defined descriptions and complications of heart disease',
        '430': 'Diseases of arteries, arterioles and capillaries',
        '431': 'Cerebral hemorrhage',
        '432': 'Other intracranial hemorrhage',
        '433': 'Occlusion and stenosis of precerebral arteries',
        '434': 'Occlusion of cerebral arteries',
        '435': 'Transient cerebral ischemia',
        '436': 'Acute, but ill-defined, cerebrovascular disease',
        '437': 'Other and ill-defined cerebrovascular disease',
        '438': 'Late effects of cerebrovascular disease',
        '440': 'Atherosclerosis',
        '441': 'Aortic aneurysm',
        '442': 'Other aneurysm',
        '443': 'Other peripheral vascular disease',
        '444': 'Arterial embolism and thrombosis',
        '445': 'Gangrene',
        '446': 'Systemic lupus erythematosus',
        '447': 'Other disorders of arteries and arterioles',
        '448': 'Diseases of capillaries',
        '450': 'Pulmonary embolism',
        '451': 'Phlebitis and thrombophlebitis',
        '452': 'Portal vein thrombosis',
        '453': 'Other venous embolism and thrombosis',
        '454': 'Varicose veins of lower extremities',
        '455': 'Hemorrhoids',
        '456': 'Varicose veins of other sites',
        '457': 'Noninfective disorders of lymphatic channels',
        '458': 'Hypotension',
        '459': 'Other disorders of circulatory system',
        '460': 'Diseases of the respiratory system',
        '461': 'Acute sinusitis',
        '462': 'Acute pharyngitis',
        '463': 'Acute tonsillitis',
        '464': 'Acute laryngitis and tracheitis',
        '465': 'Acute upper respiratory infections of multiple or unspecified sites',
        '466': 'Acute bronchitis and bronchiolitis',
        '470': 'Pneumonia and influenza',
        '471': 'Pneumonia and influenza',
        '472': 'Pneumonia and influenza',
        '473': 'Pneumonia and influenza',
        '474': 'Pneumonia and influenza',
        '475': 'Pneumonia and influenza',
        '476': 'Pneumonia and influenza',
        '477': 'Pneumonia and influenza',
        '478': 'Pneumonia and influenza',
        '479': 'Pneumonia and influenza',
        '480': 'Viral pneumonia',
        '481': 'Pneumococcal pneumonia',
        '482': 'Other bacterial pneumonia',
        '483': 'Pneumonia due to other specified organism',
        '484': 'Pneumonia in infectious diseases classified elsewhere',
        '485': 'Bronchopneumonia, organism unspecified',
        '486': 'Pneumonia, organism unspecified',
        '487': 'Influenza',
        '488': 'Influenza due to identified avian influenza virus',
        '490': 'Chronic obstructive pulmonary disease',
        '491': 'Chronic bronchitis',
        '492': 'Emphysema',
        '493': 'Asthma',
        '494': 'Bronchiectasis',
        '495': 'Extrinsic allergic alveolitis',
        '496': 'Other chronic obstructive pulmonary disease',
        '500': 'Diseases of the digestive system',
        '510': 'Diseases of the oral cavity, salivary glands and jaws',
        '520': 'Diseases of the esophagus',
        '530': 'Diseases of the stomach and duodenum',
        '540': 'Diseases of the appendix',
        '550': 'Hernia of abdominal cavity',
        '560': 'Diseases of the intestines and peritoneum',
        '570': 'Diseases of the liver',
        '571': 'Chronic liver disease and cirrhosis',
        '572': 'Abscess of liver',
        '573': 'Other disorders of liver',
        '574': 'Cholelithiasis',
        '575': 'Other disorders of gallbladder',
        '576': 'Other disorders of biliary tract',
        '577': 'Diseases of the pancreas',
        '578': 'Gastrointestinal hemorrhage',
        '579': 'Intestinal malabsorption',
        '580': 'Diseases of the genitourinary system',
        '581': 'Nephritis and nephrotic syndrome',
        '582': 'Chronic nephritis',
        '583': 'Nephritis and nephrotic syndrome',
        '584': 'Acute renal failure',
        '585': 'Chronic renal failure',
        '586': 'Renal failure, unspecified',
        '587': 'Renal sclerosis',
        '588': 'Disorders resulting from impaired renal function',
        '589': 'Small kidney of unknown cause',
        '590': 'Infections of kidney',
        '591': 'Hydronephrosis',
        '592': 'Calculus of kidney and ureter',
        '593': 'Other disorders of kidney and ureter',
        '594': 'Calculi of lower urinary tract',
        '595': 'Cystitis',
        '596': 'Other disorders of bladder',
        '597': 'Urethritis, not sexually transmitted',
        '598': 'Urethral stricture',
        '599': 'Other disorders of urethra and urinary tract',
        '600': 'Diseases of the musculoskeletal system and connective tissue',
        '610': 'Arthropathies',
        '611': 'Inflammatory polyarthropathies',
        '612': 'Other joint disorders',
        '613': 'Other joint disorders',
        '614': 'Dorsopathies',
        '615': 'Dorsopathies',
        '616': 'Rheumatism, excluding the spine',
        '617': 'Rheumatism, excluding the spine',
        '618': 'Osteopathies',
        '619': 'Osteopathies',
        '620': 'Disorders of soft tissue',
        '621': 'Disorders of soft tissue',
        '622': 'Other disorders of bone and cartilage',
        '623': 'Other disorders of bone and cartilage',
        '624': 'Osteomyelitis',
        '625': 'Osteomyelitis',
        '626': 'Osteomyelitis',
        '627': 'Osteomyelitis',
        '628': 'Osteomyelitis',
        '629': 'Osteomyelitis',
        '630': 'Disorders of the skin and subcutaneous tissue',
        '680': 'Infections of the skin and subcutaneous tissue',
        '681': 'Infections of the skin and subcutaneous tissue',
        '682': 'Infections of the skin and subcutaneous tissue',
        '683': 'Infections of the skin and subcutaneous tissue',
        '684': 'Infections of the skin and subcutaneous tissue',
        '685': 'Infections of the skin and subcutaneous tissue',
        '686': 'Other inflammatory conditions of skin',
        '690': 'Other inflammatory conditions of skin',
        '691': 'Other inflammatory conditions of skin',
        '692': 'Other inflammatory conditions of skin',
        '693': 'Dermatitis and eczema',
        '694': 'Bullous dermatoses',
        '695': 'Papulosquamous disorders',
        '696': 'Papulosquamous disorders',
        '697': 'Papulosquamous disorders',
        '698': 'Pruritus and related disorders',
        '699': 'Disorders of skin appendages',
        '700': 'Complications of pregnancy, childbirth and the puerperium',
        '710': 'Complications of pregnancy, childbirth and the puerperium',
        '720': 'Complications of pregnancy, childbirth and the puerperium',
        '730': 'Complications of pregnancy, childbirth and the puerperium',
        '740': 'Complications of pregnancy, childbirth and the puerperium',
        '750': 'Complications of pregnancy, childbirth and the puerperium',
        '760': 'Certain conditions originating in the perinatal period',
        '770': 'Congenital anomalies',
        '771': 'Congenital anomalies',
        '772': 'Congenital anomalies',
        '773': 'Congenital anomalies',
        '774': 'Congenital anomalies',
        '775': 'Congenital anomalies',
        '776': 'Congenital anomalies',
        '777': 'Congenital anomalies',
        '778': 'Congenital anomalies',
        '779': 'Congenital anomalies',
        '780': 'Symptoms, signs and ill-defined conditions',
        '781': 'Symptoms, signs and ill-defined conditions',
        '782': 'Symptoms, signs and ill-defined conditions',
        '783': 'Symptoms, signs and ill-defined conditions',
        '784': 'Symptoms, signs and ill-defined conditions',
        '785': 'Symptoms, signs and ill-defined conditions',
        '786': 'Symptoms, signs and ill-defined conditions',
        '787': 'Symptoms, signs and ill-defined conditions',
        '788': 'Symptoms, signs and ill-defined conditions',
        '789': 'Symptoms, signs and ill-defined conditions',
        '790': 'Symptoms, signs and ill-defined conditions',
        '791': 'Symptoms, signs and ill-defined conditions',
        '792': 'Symptoms, signs and ill-defined conditions',
        '793': 'Symptoms, signs and ill-defined conditions',
        '794': 'Symptoms, signs and ill-defined conditions',
        '795': 'Symptoms, signs and ill-defined conditions',
        '796': 'Symptoms, signs and ill-defined conditions',
        '797': 'Symptoms, signs and ill-defined conditions',
        '798': 'Symptoms, signs and ill-defined conditions',
        '799': 'Symptoms, signs and ill-defined conditions',
        '800': 'Injury and poisoning',
        '801': 'Injury and poisoning',
        '802': 'Injury and poisoning',
        '803': 'Injury and poisoning',
        '804': 'Injury and poisoning',
        '805': 'Injury and poisoning',
        '806': 'Injury and poisoning',
        '807': 'Injury and poisoning',
        '808': 'Injury and poisoning',
        '809': 'Injury and poisoning',
        '810': 'Injury and poisoning',
        '811': 'Injury and poisoning',
        '812': 'Injury and poisoning',
        '813': 'Injury and poisoning',
        '814': 'Injury and poisoning',
        '815': 'Injury and poisoning',
        '816': 'Injury and poisoning',
        '817': 'Injury and poisoning',
        '818': 'Injury and poisoning',
        '819': 'Injury and poisoning',
        '820': 'Injury and poisoning',
        '821': 'Injury and poisoning',
        '822': 'Injury and poisoning',
        '823': 'Injury and poisoning',
        '824': 'Injury and poisoning',
        '825': 'Injury and poisoning',
        '826': 'Injury and poisoning',
        '827': 'Injury and poisoning',
        '828': 'Injury and poisoning',
        '829': 'Injury and poisoning',
        '830': 'Injury and poisoning',
        '831': 'Injury and poisoning',
        '832': 'Injury and poisoning',
        '833': 'Injury and poisoning',
        '834': 'Injury and poisoning',
        '835': 'Injury and poisoning',
        '836': 'Injury and poisoning',
        '837': 'Injury and poisoning',
        '838': 'Injury and poisoning',
        '839': 'Injury and poisoning',
        '840': 'Injury and poisoning',
        '841': 'Injury and poisoning',
        '842': 'Injury and poisoning',
        '843': 'Injury and poisoning',
        '844': 'Injury and poisoning',
        '845': 'Injury and poisoning',
        '846': 'Injury and poisoning',
        '847': 'Injury and poisoning',
        '848': 'Injury and poisoning',
        '849': 'Injury and poisoning',
        '850': 'Injury and poisoning',
        '851': 'Injury and poisoning',
        '852': 'Injury and poisoning',
        '853': 'Injury and poisoning',
        '854': 'Injury and poisoning',
        '860': 'Injury and poisoning',
        '861': 'Injury and poisoning',
        '862': 'Injury and poisoning',
        '863': 'Injury and poisoning',
        '864': 'Injury and poisoning',
        '865': 'Injury and poisoning',
        '866': 'Injury and poisoning',
        '867': 'Injury and poisoning',
        '868': 'Injury and poisoning',
        '869': 'Injury and poisoning',
        '870': 'Injury and poisoning',
        '871': 'Injury and poisoning',
        '872': 'Injury and poisoning',
        '873': 'Injury and poisoning',
        '874': 'Injury and poisoning',
        '875': 'Injury and poisoning',
        '876': 'Injury and poisoning',
        '877': 'Injury and poisoning',
        '878': 'Injury and poisoning',
        '879': 'Injury and poisoning',
        '880': 'Injury and poisoning',
        '881': 'Injury and poisoning',
        '882': 'Injury and poisoning',
        '883': 'Injury and poisoning',
        '884': 'Injury and poisoning',
        '885': 'Injury and poisoning',
        '886': 'Injury and poisoning',
        '887': 'Injury and poisoning',
        '888': 'Injury and poisoning',
        '889': 'Injury and poisoning',
        '890': 'Injury and poisoning',
        '891': 'Injury and poisoning',
        '892': 'Injury and poisoning',
        '893': 'Injury and poisoning',
        '894': 'Injury and poisoning',
        '895': 'Injury and poisoning',
        '896': 'Injury and poisoning',
        '897': 'Injury and poisoning',
        '898': 'Injury and poisoning',
        '899': 'Injury and poisoning',
        '900': 'Injury and poisoning',
        '901': 'Injury and poisoning',
        '902': 'Injury and poisoning',
        '903': 'Injury and poisoning',
        '904': 'Injury and poisoning',
        '905': 'Injury and poisoning',
        '906': 'Injury and poisoning',
        '907': 'Injury and poisoning',
        '908': 'Injury and poisoning',
        '909': 'Injury and poisoning',
        '910': 'Injury and poisoning',
        '911': 'Injury and poisoning',
        '912': 'Injury and poisoning',
        '913': 'Injury and poisoning',
        '914': 'Injury and poisoning',
        '915': 'Injury and poisoning',
        '916': 'Injury and poisoning',
        '917': 'Injury and poisoning',
        '918': 'Injury and poisoning',
        '919': 'Injury and poisoning',
        '920': 'Injury and poisoning',
        '921': 'Injury and poisoning',
        '922': 'Injury and poisoning',
        '923': 'Injury and poisoning',
        '924': 'Injury and poisoning',
        '925': 'Injury and poisoning',
        '926': 'Injury and poisoning',
        '927': 'Injury and poisoning',
        '928': 'Injury and poisoning',
        '929': 'Injury and poisoning',
        '930': 'Injury and poisoning',
        '931': 'Injury and poisoning',
        '932': 'Injury and poisoning',
        '933': 'Injury and poisoning',
        '934': 'Injury and poisoning',
        '935': 'Injury and poisoning',
        '936': 'Injury and poisoning',
        '937': 'Injury and poisoning',
        '938': 'Injury and poisoning',
        '939': 'Injury and poisoning',
        '940': 'Injury and poisoning',
        '941': 'Injury and poisoning',
        '942': 'Injury and poisoning',
        '943': 'Injury and poisoning',
        '944': 'Injury and poisoning',
        '945': 'Injury and poisoning',
        '946': 'Injury and poisoning',
        '947': 'Injury and poisoning',
        '948': 'Injury and poisoning',
        '949': 'Injury and poisoning',
        '950': 'Injury and poisoning',
        '951': 'Injury and poisoning',
        '952': 'Injury and poisoning',
        '953': 'Injury and poisoning',
        '954': 'Injury and poisoning',
        '955': 'Injury and poisoning',
        '956': 'Injury and poisoning',
        '957': 'Injury and poisoning',
        '958': 'Injury and poisoning',
        '959': 'Injury and poisoning',
        '960': 'Poisoning',
        '961': 'Poisoning',
        '962': 'Poisoning',
        '963': 'Poisoning',
        '964': 'Poisoning',
        '965': 'Poisoning',
        '966': 'Poisoning',
        '967': 'Poisoning',
        '968': 'Poisoning',
        '969': 'Poisoning',
        '970': 'Poisoning',
        '971': 'Poisoning',
        '972': 'Poisoning',
        '973': 'Poisoning',
        '974': 'Poisoning',
        '975': 'Poisoning',
        '976': 'Poisoning',
        '977': 'Poisoning',
        '978': 'Poisoning',
        '979': 'Poisoning',
        '980': 'Poisoning',
        '981': 'Poisoning',
        '982': 'Poisoning',
        '983': 'Poisoning',
        '984': 'Poisoning',
        '985': 'Poisoning',
        '986': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '987': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '988': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '989': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '990': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '991': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '992': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '993': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '994': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '995': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '996': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '997': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '998': 'Toxic effects of substances chiefly nonmedicinal as to source',
        '999': 'Toxic effects of substances chiefly nonmedicinal as to source',
        'E': 'External causes of morbidity and mortality',
        'V': 'Supplementary classification',
    }

    # ------------------------------------------------------------------
    # Elixhauser comorbidity groups (ICD-10 → category)
    # ------------------------------------------------------------------

    ELIXHAUSER_MAP: dict[str, list[str]] = {
        'heart_failure': ['I50'],
        'diabetes': ['E10', 'E11', 'E13'],
        'copd': ['J44'],
        'renal_failure': ['N17', 'N18', 'N19'],
        'liver_disease': ['K70', 'K73', 'K74'],
        'cancer': ['C'],
        'stroke': ['I60', 'I61', 'I62', 'I63', 'I64'],
        'pneumonia': ['J12', 'J13', 'J14', 'J15', 'J18'],
        'sepsis': ['A40', 'A41'],
        'malnutrition': ['E40', 'E41', 'E42', 'E43', 'E44', 'E45', 'E46'],
    }

    # ------------------------------------------------------------------
    # ICD-9 Elixhauser mapping
    # ------------------------------------------------------------------

    ELIXHAUSER_ICD9_MAP: dict[str, list[str]] = {
        'heart_failure': ['428'],
        'diabetes': ['250'],
        'copd': ['491', '492', '496'],
        'renal_failure': ['584', '585', '586'],
        'liver_disease': ['571'],
        'cancer': ['140', '150', '160', '170', '180', '190', '200'],
        'stroke': ['430', '431', '432', '433', '434', '435', '436'],
        'pneumonia': ['480', '481', '482', '483', '484', '485', '486'],
        'sepsis': ['038'],
        'malnutrition': ['260', '261', '262', '263'],
    }

    # ------------------------------------------------------------------
    # ICD-9 disease categories (first 3 digits → category)
    # ------------------------------------------------------------------

    ICD9_CATEGORIES: dict[str, str] = {
        '250': 'Diabetes mellitus',
        '410': 'Acute myocardial infarction',
        '414': 'Chronic ischemic heart disease',
        '427': 'Cardiac dysrhythmias',
        '428': 'Heart failure',
        '430': 'Subarachnoid hemorrhage',
        '431': 'Intracerebral hemorrhage',
        '432': 'Other intracranial hemorrhage',
        '433': 'Cerebral artery occlusion',
        '434': 'Cerebral artery occlusion',
        '436': 'Acute, ill-defined cerebrovascular disease',
        '440': 'Atherosclerosis',
        '480': 'Viral pneumonia',
        '481': 'Pneumococcal pneumonia',
        '482': 'Other bacterial pneumonia',
        '485': 'Bronchopneumonia',
        '486': 'Pneumonia, organism unspecified',
        '491': 'Chronic bronchitis',
        '492': 'Emphysema',
        '493': 'Asthma',
        '496': 'Chronic obstructive pulmonary disease',
        '571': 'Chronic liver disease and cirrhosis',
        '580': 'Acute nephritis',
        '584': 'Acute renal failure',
        '585': 'Chronic renal failure',
        '586': 'Renal failure, unspecified',
        '707': 'Chronic ulcer of skin',
        '785': 'Cardiovascular symptoms',
        '786': 'Respiratory symptoms',
    }

    def __init__(self) -> None:
        """Initialize ICDGrouper with logger."""
        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Version detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_version(icd_code: str) -> int:
        """Detect ICD version from code format.

        ICD-10 codes start with a letter followed by 2-4 alphanumeric chars.
        ICD-9 codes are numeric, optionally with a decimal point.

        Parameters
        ----------
        icd_code : str
            The ICD code to classify.

        Returns
        -------
        int
            9 or 10.
        """
        code = str(icd_code).strip().upper()
        if not code:
            return 10
        if code[0].isalpha():
            return 10
        return 9

    # ------------------------------------------------------------------
    # Core grouping methods
    # ------------------------------------------------------------------

    def get_chapter(self, icd_code: str) -> str:
        """Get ICD-10 chapter from code.

        Parameters
        ----------
        icd_code : str
            ICD-10 code (e.g., ``'I50.9'``).

        Returns
        -------
        str
            Chapter description string.

        SQL equivalent::

            SELECT CASE
                WHEN LEFT(icd_code, 1) IN ('A','B') THEN
                    'Certain infectious and parasitic diseases'
                WHEN LEFT(icd_code, 1) = 'I' THEN
                    'Diseases of the circulatory system'
                ...
            END AS chapter
        """
        code = str(icd_code).strip().upper()
        if not code:
            return 'Unknown chapter'

        version = self.detect_version(code)
        if version == 10:
            return self.ICD10_CHAPTERS.get(code[0], 'Unknown chapter')

        # ICD-9: use first 3 digits for chapter lookup
        digits = re.sub(r'[^0-9]', '', code)
        prefix3 = digits[:3] if len(digits) >= 3 else digits
        return self.ICD9_CHAPTERS.get(prefix3, 'Unknown chapter')

    def get_category(self, icd_code: str) -> str:
        """Get detailed disease category from ICD code.

        Parameters
        ----------
        icd_code : str
            ICD code.

        Returns
        -------
        str
            Category description.

        SQL equivalent::

            SELECT CASE
                WHEN icd_code LIKE 'I50%' THEN 'Heart failure'
                WHEN icd_code LIKE 'E11%' THEN 'Type 2 diabetes mellitus'
                WHEN icd_code LIKE 'J44%' THEN 'Chronic obstructive pulmonary disease'
                ...
            END AS category
        """
        code = str(icd_code).strip().upper()
        if not code:
            return 'Unknown category'

        version = self.detect_version(code)

        # ICD-10: first 3 characters (letter + 2 digits) define category
        if version == 10:
            prefix = code[:3]
            # Build on-the-fly from ELIXHAUSER_MAP for common conditions
            for group, prefixes in self.ELIXHAUSER_MAP.items():
                for p in prefixes:
                    if code.startswith(p):
                        return group.replace('_', ' ').title()
            return f'ICD-10 category {prefix}'

        # ICD-9: first 3 digits
        digits = re.sub(r'[^0-9]', '', code)
        prefix3 = digits[:3] if len(digits) >= 3 else digits
        return self.ICD9_CATEGORIES.get(prefix3, f'ICD-9 category {prefix3}')

    def get_elixhauser_group(self, icd_code: str) -> str:
        """Map ICD code to Elixhauser comorbidity group.

        Parameters
        ----------
        icd_code : str
            ICD-9 or ICD-10 code.

        Returns
        -------
        str
            Comorbidity group name or ``'none'``.

        SQL equivalent::

            SELECT CASE
                WHEN LEFT(icd_code, 3) IN ('I50') THEN 'heart_failure'
                WHEN LEFT(icd_code, 3) IN ('E10','E11','E13') THEN 'diabetes'
                WHEN LEFT(icd_code, 3) IN ('J44') THEN 'copd'
                WHEN LEFT(icd_code, 3) IN ('N17','N18','N19') THEN 'renal_failure'
                WHEN LEFT(icd_code, 3) IN ('K70','K73','K74') THEN 'liver_disease'
                WHEN LEFT(icd_code, 1) = 'C' THEN 'cancer'
                WHEN LEFT(icd_code, 3) IN ('I60','I61','I62','I63','I64') THEN 'stroke'
                WHEN LEFT(icd_code, 3) IN ('J12','J13','J14','J15','J18') THEN 'pneumonia'
                WHEN LEFT(icd_code, 3) IN ('A40','A41') THEN 'sepsis'
                WHEN LEFT(icd_code, 3) IN ('E40','E41','E42','E43','E44','E45','E46')
                    THEN 'malnutrition'
                ELSE 'none'
            END AS elixhauser_group
        """
        code = str(icd_code).strip().upper()
        if not code:
            return 'none'

        version = self.detect_version(code)
        mapping = self.ELIXHAUSER_ICD9_MAP if version == 9 else self.ELIXHAUSER_MAP

        for group, prefixes in mapping.items():
            for prefix in prefixes:
                if code.startswith(prefix):
                    return group

        return 'none'

    # ------------------------------------------------------------------
    # DataFrame operations (vectorized)
    # ------------------------------------------------------------------

    def group_icd_codes(
        self, df: pd.DataFrame, code_column: str = 'icd_code'
    ) -> pd.DataFrame:
        """Add chapter, category, and Elixhauser group columns to DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing ICD codes.
        code_column : str
            Name of the column with ICD codes.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with three new columns added.
        """
        if code_column not in df.columns:
            raise ValueError(f"Column '{code_column}' not found in DataFrame.")

        df = df.copy()

        # Vectorized chapter mapping
        df['icd_chapter'] = df[code_column].map(self.get_chapter)
        df['icd_category'] = df[code_column].map(self.get_category)
        df['elixhauser_group'] = df[code_column].map(self.get_elixhauser_group)

        logger.info(
            "Grouped %d ICD codes into %d unique chapters, %d categories, "
            "and %d Elixhauser groups.",
            len(df),
            df['icd_chapter'].nunique(),
            df['icd_category'].nunique(),
            df['elixhauser_group'].nunique(),
        )
        return df

    def diagnosis_frequency_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank diagnoses by frequency.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least one ICD code column.  The method scans
            for columns named ``icd_code``, ``diagnosis_code``, or
            ``primary_diagnosis_code`` in order of preference.

        Returns
        -------
        pd.DataFrame
            Columns: icd_code, description, count, rank, cumulative_pct
        """
        # Auto-detect the ICD code column
        code_col = None
        for candidate in ('icd_code', 'diagnosis_code', 'primary_diagnosis_code'):
            if candidate in df.columns:
                code_col = candidate
                break
        if code_col is None:
            raise ValueError(
                "No ICD code column found. Expected 'icd_code', "
                "'diagnosis_code', or 'primary_diagnosis_code'."
            )

        counts = (
            df[code_col]
            .value_counts()
            .reset_index()
        )
        counts.columns = ['icd_code', 'count']
        counts['rank'] = range(1, len(counts) + 1)
        counts['description'] = counts['icd_code'].map(self.get_category)
        total = counts['count'].sum()
        counts['cumulative_pct'] = (counts['count'].cumsum() / total * 100).round(2)

        return counts[['icd_code', 'description', 'count', 'rank', 'cumulative_pct']]

    def comorbidity_count(
        self, df: pd.DataFrame, patient_id_column: str = 'patient_id'
    ) -> pd.DataFrame:
        """Count comorbidities per patient based on Elixhauser grouping.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with patient and ICD code columns.  Scans for ICD
            code column among ``icd_code``, ``diagnosis_code``,
            ``primary_diagnosis_code``.
        patient_id_column : str
            Name of the patient identifier column.

        Returns
        -------
        pd.DataFrame
            Columns: patient_id, comorbidity_count, comorbidity_groups
        """
        # Auto-detect ICD code column
        code_col = None
        for candidate in ('icd_code', 'diagnosis_code', 'primary_diagnosis_code'):
            if candidate in df.columns:
                code_col = candidate
                break
        if code_col is None:
            raise ValueError(
                "No ICD code column found. Expected 'icd_code', "
                "'diagnosis_code', or 'primary_diagnosis_code'."
            )

        temp = df[[patient_id_column, code_col]].copy()
        temp['elixhauser_group'] = temp[code_col].map(self.get_elixhauser_group)
        temp = temp[temp['elixhauser_group'] != 'none']

        result = (
            temp.groupby(patient_id_column)
            .agg(
                comorbidity_count=('elixhauser_group', 'nunique'),
                comorbidity_groups=('elixhauser_group', lambda x: ', '.join(sorted(set(x)))),
            )
            .reset_index()
        )

        logger.info(
            "Computed comorbidity counts for %d patients.", len(result)
        )
        return result

    # ------------------------------------------------------------------
    # SQL CASE WHEN documentation
    # ------------------------------------------------------------------

    def case_hierarchy_sql(self) -> str:
        """Return CASE WHEN SQL statement for ICD grouping.

        This is a ready-to-use SQL snippet for database-side ICD grouping.
        It covers both ICD-10 chapters and Elixhauser comorbidity mapping.

        Returns
        -------
        str
            Multi-line SQL string.

        Example usage in a query::

            SELECT
                patient_id,
                icd_code,
                (<this_output>) AS chapter
            FROM diagnoses
        """
        return """
-- ICD-10 Chapter Mapping via CASE Hierarchy
-- ==========================================
-- Maps ICD-10 codes to WHO chapter descriptions based on the first character.
--
SELECT
    icd_code,
    CASE
        WHEN LEFT(icd_code, 1) IN ('A','B') THEN
            'Certain infectious and parasitic diseases'
        WHEN LEFT(icd_code, 1) = 'C' THEN
            'Neoplasms'
        WHEN LEFT(icd_code, 1) = 'D' THEN
            'Diseases of the blood and blood-forming organs'
        WHEN LEFT(icd_code, 1) = 'E' THEN
            'Endocrine, nutritional and metabolic diseases'
        WHEN LEFT(icd_code, 1) = 'F' THEN
            'Mental and behavioural disorders'
        WHEN LEFT(icd_code, 1) = 'G' THEN
            'Diseases of the nervous system'
        WHEN LEFT(icd_code, 1) = 'H' THEN
            'Diseases of the eye and adnexa / ear'
        WHEN LEFT(icd_code, 1) = 'I' THEN
            'Diseases of the circulatory system'
        WHEN LEFT(icd_code, 1) = 'J' THEN
            'Diseases of the respiratory system'
        WHEN LEFT(icd_code, 1) = 'K' THEN
            'Diseases of the digestive system'
        WHEN LEFT(icd_code, 1) = 'L' THEN
            'Diseases of the skin and subcutaneous tissue'
        WHEN LEFT(icd_code, 1) = 'M' THEN
            'Diseases of the musculoskeletal system and connective tissue'
        WHEN LEFT(icd_code, 1) = 'N' THEN
            'Diseases of the genitourinary system'
        WHEN LEFT(icd_code, 1) = 'O' THEN
            'Pregnancy, childbirth and the puerperium'
        WHEN LEFT(icd_code, 1) = 'P' THEN
            'Certain conditions originating in the perinatal period'
        WHEN LEFT(icd_code, 1) = 'Q' THEN
            'Congenital malformations, deformations and chromosomal abnormalities'
        WHEN LEFT(icd_code, 1) = 'R' THEN
            'Symptoms, signs and abnormal clinical and laboratory findings'
        WHEN LEFT(icd_code, 1) IN ('S','T') THEN
            'Injury, poisoning and certain other consequences of external causes'
        WHEN LEFT(icd_code, 1) IN ('V','W','X','Y') THEN
            'External causes of morbidity and mortality'
        WHEN LEFT(icd_code, 1) = 'Z' THEN
            'Factors influencing health status and contact with health services'
        ELSE 'Unknown chapter'
    END AS chapter,

-- Elixhauser Comorbidity Mapping via CASE Hierarchy
-- ==================================================
-- Maps ICD-10 codes to Elixhauser comorbidity groups for risk adjustment.
--
    CASE
        WHEN LEFT(icd_code, 3) IN ('I50') THEN 'heart_failure'
        WHEN LEFT(icd_code, 3) IN ('E10','E11','E13') THEN 'diabetes'
        WHEN LEFT(icd_code, 3) IN ('J44') THEN 'copd'
        WHEN LEFT(icd_code, 3) IN ('N17','N18','N19') THEN 'renal_failure'
        WHEN LEFT(icd_code, 3) IN ('K70','K73','K74') THEN 'liver_disease'
        WHEN LEFT(icd_code, 1) = 'C' THEN 'cancer'
        WHEN LEFT(icd_code, 3) IN ('I60','I61','I62','I63','I64') THEN 'stroke'
        WHEN LEFT(icd_code, 3) IN ('J12','J13','J14','J15','J18') THEN 'pneumonia'
        WHEN LEFT(icd_code, 3) IN ('A40','A41') THEN 'sepsis'
        WHEN LEFT(icd_code, 3) IN ('E40','E41','E42','E43','E44','E45','E46')
            THEN 'malnutrition'
        ELSE 'none'
    END AS elixhauser_group
FROM diagnoses
ORDER BY icd_code;
""".strip()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def grouping_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Summary of diagnosis groupings with counts and percentages.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with at least one ICD code column.  If chapter/
            category/Elixhauser columns already exist, they are reused;
            otherwise ``group_icd_codes()`` is called automatically.

        Returns
        -------
        pd.DataFrame
            Columns: group_type, group_name, count, percentage
        """
        df = df.copy()

        # Ensure grouping columns exist
        code_col = None
        for candidate in ('icd_code', 'diagnosis_code', 'primary_diagnosis_code'):
            if candidate in df.columns:
                code_col = candidate
                break

        if code_col is None:
            raise ValueError(
                "No ICD code column found. Expected 'icd_code', "
                "'diagnosis_code', or 'primary_diagnosis_code'."
            )

        if 'icd_chapter' not in df.columns:
            df = self.group_icd_codes(df, code_col)

        total = len(df)

        summaries = []
        for group_col, group_type in [
            ('icd_chapter', 'chapter'),
            ('icd_category', 'category'),
            ('elixhauser_group', 'elixhauser_group'),
        ]:
            counts = df[group_col].value_counts().reset_index()
            counts.columns = ['group_name', 'count']
            counts['group_type'] = group_type
            counts['percentage'] = (counts['count'] / total * 100).round(2)
            summaries.append(counts)

        result = pd.concat(summaries, ignore_index=True)
        return result[['group_type', 'group_name', 'count', 'percentage']]
