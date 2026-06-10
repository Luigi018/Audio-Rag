"""Static script definitions for synthetic audio dataset generation.

All 25 scripts are hardcoded here for full reproducibility.
Topics intentionally overlap across three pairs to test multi-document retrieval:
  - "technology"  : it_m_002 + en_m_001
  - "economics"   : it_f_005 + en_m_004
  - "sport"       : it_f_006 + it_m_004
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AudioScript:
    script_id: str
    text: str
    language: str           # "it" | "en"
    gender: str             # "male" | "female"
    voice: str              # Kokoro voice identifier
    topic: str
    expected_keywords: list[str]
    filename: str           # e.g. "it_female_001_politica.wav"


# ---------------------------------------------------------------------------
# Italian — Female (voice: if_sara)
# ---------------------------------------------------------------------------

_IT_F_001 = AudioScript(
    script_id="it_f_001_politica",
    filename="it_female_001_politica.wav",
    language="it",
    gender="female",
    voice="if_sara",
    topic="politica_italiana",
    expected_keywords=["politica italiana", "governo", "parlamento", "partiti", "elezioni"],
    text=(
        "La politica italiana è caratterizzata da una forte frammentazione partitica che rende "
        "difficile la formazione di governi stabili. Negli ultimi decenni si sono succedute "
        "numerose coalizioni di centrodestra e centrosinistra, con frequenti cambi di esecutivo. "
        "Il sistema bicamerale perfetto, con Camera e Senato dotati degli stessi poteri, è spesso "
        "citato come una delle cause della lentezza del processo legislativo. Le riforme "
        "istituzionali sono da anni al centro del dibattito politico nazionale. Il sistema "
        "elettorale ha subito numerose modifiche nel corso degli anni, passando da sistemi "
        "proporzionali a sistemi misti, con effetti significativi sulla governabilità del paese. "
        "La partecipazione dei cittadini alla vita politica rimane una sfida importante, con "
        "tassi di astensione in crescita nelle ultime elezioni."
    ),
)

_IT_F_002 = AudioScript(
    script_id="it_f_002_cucina",
    filename="it_female_002_cucina.wav",
    language="it",
    gender="female",
    voice="if_sara",
    topic="cucina_italiana",
    expected_keywords=["cucina italiana", "gastronomia", "piatti tipici", "ingredienti", "ricette"],
    text=(
        "La cucina italiana è considerata una delle più ricche e variegate al mondo, con una "
        "tradizione gastronomica che affonda le radici in secoli di storia e cultura regionale. "
        "Ogni regione italiana vanta piatti tipici unici: dalla pasta al ragù bolognese "
        "dell'Emilia-Romagna alla pizza napoletana della Campania, dalla ribollita toscana al "
        "risotto alla milanese del Nord. L'utilizzo di ingredienti freschi e di stagione è un "
        "principio fondamentale della cucina tradizionale italiana. L'olio d'oliva extravergine, "
        "il pomodoro, la mozzarella e il parmigiano reggiano sono tra i simboli gastronomici più "
        "riconoscibili nel mondo. La Dieta Mediterranea, di cui la cucina italiana è parte "
        "integrante, è stata riconosciuta come Patrimonio Culturale Immateriale dell'Umanità "
        "dall'UNESCO."
    ),
)

_IT_F_003 = AudioScript(
    script_id="it_f_003_arte",
    filename="it_female_003_arte.wav",
    language="it",
    gender="female",
    voice="if_sara",
    topic="arte_rinascimentale",
    expected_keywords=["arte rinascimentale", "Leonardo da Vinci", "Michelangelo", "pittura", "Rinascimento"],
    text=(
        "Il Rinascimento italiano, fiorito tra il XIV e il XVI secolo, rappresenta uno dei momenti "
        "più straordinari nella storia dell'arte mondiale. Firenze fu il principale centro di questa "
        "rivoluzione artistica e culturale, grazie al mecenatismo della famiglia Medici. Artisti "
        "come Leonardo da Vinci, Michelangelo e Raffaello ridefinirono i canoni della pittura, "
        "della scultura e dell'architettura, introducendo tecniche innovative come la prospettiva "
        "lineare e lo sfumato. La Cappella Sistina a Roma, con il celebre affresco del soffitto di "
        "Michelangelo, è considerata uno dei capolavori assoluti dell'arte rinascimentale. Il David "
        "di Michelangelo e La Gioconda di Leonardo da Vinci sono tra le opere più famose e ammirate "
        "al mondo, simboli immortali del genio artistico italiano."
    ),
)

_IT_F_004 = AudioScript(
    script_id="it_f_004_turismo",
    filename="it_female_004_turismo.wav",
    language="it",
    gender="female",
    voice="if_sara",
    topic="turismo_italia",
    expected_keywords=["turismo", "Italia", "patrimonio UNESCO", "città d'arte", "viaggi"],
    text=(
        "L'Italia è una delle mete turistiche più visitate al mondo, con oltre sessanta milioni di "
        "turisti stranieri ogni anno. Il paese possiede il maggior numero di siti patrimonio UNESCO "
        "al mondo, tra cui il centro storico di Roma, Venezia con la sua laguna, i trulli di "
        "Alberobello e le Cinque Terre in Liguria. Roma, Firenze, Venezia e Milano sono le città "
        "più visitate, ciascuna con un'offerta culturale e artistica incomparabile. Il turismo "
        "enogastronomico è in forte crescita, con visitatori che scelgono l'Italia per degustare "
        "i suoi vini, formaggi, salumi e piatti regionali. Le coste italiane, dal Mar Tirreno "
        "all'Adriatico, offrono alcune delle spiagge più belle del Mediterraneo, attirando milioni "
        "di turisti balneari ogni estate."
    ),
)

_IT_F_005 = AudioScript(
    script_id="it_f_005_economia",
    filename="it_female_005_economia.wav",
    language="it",
    gender="female",
    voice="if_sara",
    topic="economics",
    expected_keywords=["economia italiana", "PIL", "imprese", "debito pubblico", "manifattura"],
    text=(
        "L'economia italiana è la terza più grande dell'Eurozona e la seconda manifatturiera "
        "d'Europa. Il tessuto produttivo italiano è caratterizzato da un elevato numero di piccole "
        "e medie imprese, molte delle quali operano in distretti industriali specializzati. Il "
        "settore della moda, del lusso e del design ha una vocazione fortemente internazionale, con "
        "marchi come Ferrari, Gucci, Prada e Armani riconosciuti in tutto il mondo. Il debito "
        "pubblico elevato rimane una delle principali criticità dell'economia italiana, con un "
        "rapporto debito PIL tra i più alti nell'Unione Europea. Il Meridione del paese presenta "
        "storicamente un livello di sviluppo economico inferiore rispetto al Nord, con tassi di "
        "disoccupazione più elevati e un PIL pro capite più basso."
    ),
)

_IT_F_006 = AudioScript(
    script_id="it_f_006_sport",
    filename="it_female_006_sport.wav",
    language="it",
    gender="female",
    voice="if_sara",
    topic="sport",
    expected_keywords=["sport", "Italia", "calcio", "campionati", "medaglie"],
    text=(
        "Lo sport occupa un ruolo centrale nella cultura italiana, con una passione che attraversa "
        "tutte le generazioni. Il calcio è senza dubbio lo sport più amato, con milioni di tifosi "
        "appassionati e squadre di club tra le più famose al mondo come Juventus, Inter e Milan. "
        "L'Italia ha conquistato quattro titoli mondiali di calcio nel 1934, 1938, 1982 e 2006, "
        "oltre al campionato europeo nel 2021. Lo sci alpino, la Formula 1 e il ciclismo sono "
        "altri sport in cui l'Italia eccelle a livello internazionale. La pallavolo e la "
        "pallacanestro godono di grande popolarità, con nazionali competitive e leghe "
        "professionistiche di alto livello. Gli Azzurri, come vengono comunemente chiamate le "
        "nazionali italiane di vari sport, rappresentano un simbolo di identità nazionale."
    ),
)

# ---------------------------------------------------------------------------
# Italian — Male (voice: im_nicola)
# ---------------------------------------------------------------------------

_IT_M_001 = AudioScript(
    script_id="it_m_001_storia",
    filename="it_male_001_storia.wav",
    language="it",
    gender="male",
    voice="im_nicola",
    topic="storia_italiana",
    expected_keywords=["storia italiana", "Roma antica", "Risorgimento", "Unità d'Italia", "Repubblica"],
    text=(
        "La storia italiana abbraccia millenni di civiltà, dall'antica Roma all'era moderna. La "
        "civiltà romana, che raggiunse il suo apice nel primo e secondo secolo dopo Cristo, lasciò "
        "un'eredità culturale, giuridica e architettonica che ancora oggi permea la cultura "
        "occidentale. Dopo la caduta dell'Impero Romano d'Occidente nel 476, la penisola italiana "
        "fu teatro di invasioni barbariche, domini stranieri e la nascita delle città stato "
        "medievali. Il Risorgimento, il movimento di unificazione nazionale del diciannovesimo "
        "secolo, portò alla fondazione del Regno d'Italia nel 1861 grazie all'opera di figure "
        "storiche come Giuseppe Garibaldi, Camillo Cavour e Vittorio Emanuele Secondo. Il "
        "ventesimo secolo vide l'Italia protagonista di due guerre mondiali, un ventennio fascista "
        "e la rinascita democratica con la proclamazione della Repubblica nel 1946."
    ),
)

_IT_M_002 = AudioScript(
    script_id="it_m_002_tecnologia",
    filename="it_male_002_tecnologia.wav",
    language="it",
    gender="male",
    voice="im_nicola",
    topic="technology",
    expected_keywords=["tecnologia", "innovazione", "intelligenza artificiale", "startup", "digitale"],
    text=(
        "L'Italia vanta una lunga tradizione di innovazione tecnologica, dalle invenzioni "
        "rinascimentali di Leonardo da Vinci alle moderne eccellenze nel design industriale e "
        "nell'automazione. Il paese ha dato i natali a importanti pionieri della scienza e della "
        "tecnologia: Guglielmo Marconi inventò la radio, Alessandro Volta la pila elettrica ed "
        "Enrico Fermi contribuì in modo determinante allo sviluppo della fisica nucleare. Oggi "
        "l'ecosistema delle startup tecnologiche italiane è in forte crescita, con hub di "
        "innovazione a Milano, Roma e Torino. L'industria manifatturiera italiana sta attraversando "
        "una profonda trasformazione digitale, con l'adozione di tecnologie come l'intelligenza "
        "artificiale, l'Internet delle Cose e la robotica avanzata. Il Piano Nazionale di Ripresa "
        "e Resilienza prevede ingenti investimenti nella trasformazione digitale del paese."
    ),
)

_IT_M_003 = AudioScript(
    script_id="it_m_003_ambiente",
    filename="it_male_003_ambiente.wav",
    language="it",
    gender="male",
    voice="im_nicola",
    topic="ambiente",
    expected_keywords=["ambiente", "cambiamenti climatici", "energia rinnovabile", "inquinamento", "biodiversità"],
    text=(
        "L'Italia affronta sfide ambientali significative legate ai cambiamenti climatici, "
        "all'inquinamento e alla gestione delle risorse naturali. Il paese è particolarmente "
        "vulnerabile agli effetti del riscaldamento globale: ondate di calore sempre più intense, "
        "siccità prolungate nel Meridione e alluvioni catastrofiche al Nord rappresentano emergenze "
        "sempre più frequenti. La biodiversità italiana è eccezionale, con numerose specie endemiche "
        "che trovano rifugio nei parchi nazionali e nelle riserve naturali. Il settore delle energie "
        "rinnovabili è in forte espansione, con l'energia solare e quella eolica che coprono una "
        "quota crescente del fabbisogno energetico nazionale. La gestione dei rifiuti rimane una "
        "problematica irrisolta, con forti disparità tra le regioni settentrionali, dove la raccolta "
        "differenziata è avanzata, e alcune aree meridionali."
    ),
)

_IT_M_004 = AudioScript(
    script_id="it_m_004_calcio",
    filename="it_male_004_calcio.wav",
    language="it",
    gender="male",
    voice="im_nicola",
    topic="sport",
    expected_keywords=["calcio", "Serie A", "Nazionale azzurra", "campionati mondiali", "squadre"],
    text=(
        "Il calcio italiano ha una storia gloriosa che lo rende uno dei più seguiti e rispettati "
        "al mondo. La Serie A, il massimo campionato professionistico italiano, ospita alcune delle "
        "squadre più blasonate d'Europa, tra cui la Juventus di Torino, l'Inter e il Milan di "
        "Milano e la Roma e la Lazio della capitale. Il campionato si distingue per un'elevata "
        "tattica e tecnica di gioco, con allenatori italiani riconosciuti come tra i migliori al "
        "mondo. La Nazionale azzurra ha conquistato quattro Coppe del Mondo e un Campionato "
        "Europeo, consolidando il primato dell'Italia nel panorama calcistico internazionale. "
        "I derby cittadini, come il Derby della Madonnina tra Inter e Milan e il Derby della "
        "Capitale tra Roma e Lazio, sono eventi sportivi e sociali di straordinaria intensità "
        "emotiva."
    ),
)

_IT_M_005 = AudioScript(
    script_id="it_m_005_musica",
    filename="it_male_005_musica.wav",
    language="it",
    gender="male",
    voice="im_nicola",
    topic="musica_classica",
    expected_keywords=["musica classica", "opera lirica", "Verdi", "Puccini", "bel canto"],
    text=(
        "L'Italia ha dato un contributo fondamentale alla storia della musica classica occidentale. "
        "Compositori come Antonio Vivaldi, Claudio Monteverdi, Luigi Boccherini e Gioacchino "
        "Rossini hanno plasmato lo sviluppo della musica europea per secoli. L'opera lirica nacque "
        "in Italia alla fine del sedicesimo secolo come fusione di musica, dramma e arte scenica, "
        "e compositori come Giuseppe Verdi e Giacomo Puccini la portarono a vette di assoluta "
        "perfezione artistica. Teatri come la Scala di Milano, il San Carlo di Napoli e la Fenice "
        "di Venezia sono considerati tra i templi mondiali della musica lirica. Il bel canto "
        "italiano, caratterizzato da una tecnica vocale raffinata e da una grande attenzione "
        "all'espressività melodica, continua a esercitare la sua influenza nella formazione dei "
        "cantanti lirici di tutto il mondo."
    ),
)

_IT_M_006 = AudioScript(
    script_id="it_m_006_cinema",
    filename="it_male_006_cinema.wav",
    language="it",
    gender="male",
    voice="im_nicola",
    topic="cinema_italiano",
    expected_keywords=["cinema italiano", "neorealismo", "Fellini", "Visconti", "registi"],
    text=(
        "Il cinema italiano ha attraversato stagioni di straordinaria creatività artistica, "
        "lasciando un'impronta indelebile nella storia del cinema mondiale. Il neorealismo, "
        "sviluppatosi nell'immediato dopoguerra con registi come Roberto Rossellini, Vittorio "
        "De Sica e Luchino Visconti, rivoluzionò il linguaggio cinematografico con la sua "
        "rappresentazione cruda e autentica della realtà sociale italiana. Federico Fellini, con "
        "capolavori come La dolce vita, Amarcord e Otto e mezzo, portò il cinema italiano a un "
        "livello di astrazione poetica e autobiografica senza precedenti. Sergio Leone ridefinì il "
        "genere western con la sua trilogia del dollaro, creando un'estetica originale che "
        "influenzò generazioni di registi in tutto il mondo. Oggi Paolo Sorrentino e Matteo Garrone "
        "continuano la tradizione del cinema d'autore italiano con riconoscimenti internazionali di "
        "primo piano."
    ),
)

_IT_M_007 = AudioScript(
    script_id="it_m_007_scienza",
    filename="it_male_007_scienza.wav",
    language="it",
    gender="male",
    voice="im_nicola",
    topic="scienza",
    expected_keywords=["scienza", "Galileo Galilei", "Fermi", "fisica", "ricerca scientifica"],
    text=(
        "La storia della scienza italiana è costellata di scoperte e innovazioni che hanno cambiato "
        "il corso della conoscenza umana. Galileo Galilei, con le sue osservazioni astronomiche e "
        "i suoi esperimenti sul moto dei corpi, pose le fondamenta del metodo scientifico moderno. "
        "Leonardo da Vinci, oltre che artista di genio, fu un instancabile sperimentatore e "
        "inventore, anticipando concetti e macchine che si sarebbero realizzati secoli dopo. "
        "Nell'era moderna, l'Italia ha prodotto scienziati di fama mondiale: Enrico Fermi ottenne "
        "il Premio Nobel per la fisica nel 1938 e contribuì allo sviluppo della prima pila atomica. "
        "La ricerca scientifica italiana è attiva in numerosi campi, dall'astrofisica alla biologia "
        "molecolare, dalla fisica delle particelle alla medicina, con istituzioni come il CERN e "
        "l'Istituto Nazionale di Fisica Nucleare tra i protagonisti della ricerca globale."
    ),
)

# ---------------------------------------------------------------------------
# English — Female (af_heart for 1-3, af_bella for 4-6)
# ---------------------------------------------------------------------------

_EN_F_001 = AudioScript(
    script_id="en_f_001_politics",
    filename="en_female_001_politics.wav",
    language="en",
    gender="female",
    voice="af_heart",
    topic="us_politics",
    expected_keywords=["US politics", "Congress", "political parties", "elections", "polarization"],
    text=(
        "American politics has long been characterized by its two-party system, with the Democratic "
        "and Republican parties dominating the political landscape at both federal and state levels. "
        "The separation of powers between the executive, legislative, and judicial branches is a "
        "cornerstone of the United States government, designed to prevent any single branch from "
        "accumulating excessive power. Presidential elections, held every four years, attract "
        "enormous public attention and international interest, often determining the direction of "
        "domestic and foreign policy for years to come. The polarization of American society has "
        "increased significantly in recent decades, with voters becoming more ideologically sorted "
        "along party lines. Issues such as immigration, healthcare, gun control, and climate policy "
        "remain deeply divisive, making bipartisan cooperation increasingly difficult in Congress."
    ),
)

_EN_F_002 = AudioScript(
    script_id="en_f_002_travel",
    filename="en_female_002_travel.wav",
    language="en",
    gender="female",
    voice="af_heart",
    topic="travel",
    expected_keywords=["travel", "tourism", "cultural exploration", "destinations", "sustainable tourism"],
    text=(
        "Traveling is one of the most enriching experiences a person can have, offering opportunities "
        "to discover new cultures, languages, and ways of life. Whether exploring ancient ruins in "
        "Greece, surfing on beaches in Australia, or wandering through the bustling markets of "
        "Morocco, every destination offers unique insights into the diversity of human experience. "
        "Budget travel has become increasingly accessible through platforms that connect travelers "
        "with local hosts and affordable accommodation options. Sustainable tourism is gaining "
        "momentum as travelers become more aware of their environmental impact, choosing eco-friendly "
        "lodgings and supporting local businesses. Solo travel, once considered unconventional, has "
        "grown in popularity, empowering individuals to explore the world on their own terms and "
        "at their own pace, fostering independence and self-discovery."
    ),
)

_EN_F_003 = AudioScript(
    script_id="en_f_003_health",
    filename="en_female_003_health.wav",
    language="en",
    gender="female",
    voice="af_heart",
    topic="health_wellness",
    expected_keywords=["health", "wellness", "physical activity", "diet", "mental health"],
    text=(
        "Health and wellness have become central concerns for people around the world as medical "
        "research continues to reveal the profound connections between lifestyle choices and "
        "long-term health outcomes. Regular physical activity, a balanced diet rich in fruits, "
        "vegetables, and whole grains, and adequate sleep are consistently identified as the most "
        "important pillars of good health. Mental health has gained increasing recognition as "
        "equally important as physical health, with growing awareness of conditions like depression, "
        "anxiety, and burnout. Preventive medicine, which focuses on maintaining health and "
        "preventing disease before it occurs, is increasingly seen as more cost-effective and "
        "humane than treating illness after the fact. Mindfulness practices, including meditation "
        "and yoga, have moved from niche spiritual disciplines to mainstream wellness tools, "
        "supported by substantial scientific evidence of their benefits."
    ),
)

_EN_F_004 = AudioScript(
    script_id="en_f_004_climate",
    filename="en_female_004_climate.wav",
    language="en",
    gender="female",
    voice="af_bella",
    topic="climate_change",
    expected_keywords=["climate change", "global warming", "renewable energy", "carbon emissions", "Paris Agreement"],
    text=(
        "Climate change represents one of the most urgent and complex challenges facing humanity in "
        "the twenty-first century. Scientific consensus is overwhelming: human activities, "
        "particularly the burning of fossil fuels and deforestation, are driving a rapid increase "
        "in global temperatures with potentially catastrophic consequences. Rising sea levels "
        "threaten coastal communities worldwide, while more frequent and intense extreme weather "
        "events — hurricanes, droughts, wildfires, and floods — are already causing significant "
        "economic and humanitarian damage. The Paris Agreement, signed by nearly two hundred "
        "countries, established the goal of limiting global warming to well below two degrees "
        "Celsius above pre-industrial levels. Transitioning to renewable energy sources such as "
        "solar, wind, and hydroelectric power is considered essential for reducing greenhouse gas "
        "emissions and achieving carbon neutrality by mid-century."
    ),
)

_EN_F_005 = AudioScript(
    script_id="en_f_005_literature",
    filename="en_female_005_literature.wav",
    language="en",
    gender="female",
    voice="af_bella",
    topic="literature",
    expected_keywords=["literature", "books", "reading", "fiction", "authors"],
    text=(
        "Literature has served as both a mirror and a lamp throughout human history, reflecting "
        "the world as it is while illuminating possibilities for how it might be. The Western "
        "literary canon, from Homer's Iliad and Odyssey through Shakespeare's plays to the modern "
        "novels of the twentieth century, traces the evolution of human thought, emotion, and "
        "social organization across millennia. Contemporary literature has expanded dramatically "
        "to include voices from previously marginalized communities, enriching the global "
        "conversation with diverse perspectives and experiences. The rise of digital publishing "
        "and self-publishing platforms has democratized access to readership for aspiring writers, "
        "bypassing traditional gatekeepers. Reading fiction has been shown to develop empathy, "
        "improve vocabulary, and enhance critical thinking skills, making literature not merely an "
        "aesthetic pursuit but a cognitive and social one."
    ),
)

_EN_F_006 = AudioScript(
    script_id="en_f_006_education",
    filename="en_female_006_education.wav",
    language="en",
    gender="female",
    voice="af_bella",
    topic="education",
    expected_keywords=["education", "schools", "learning", "teachers", "students"],
    text=(
        "Education is widely recognized as the foundation of individual opportunity and social "
        "progress, yet significant disparities in educational access and quality persist both "
        "within and between countries. The traditional model of education, centered on classroom "
        "instruction and standardized testing, is being challenged by new approaches that emphasize "
        "critical thinking, creativity, collaboration, and communication. Technology has transformed "
        "educational delivery, with online learning platforms making knowledge accessible to "
        "students around the world regardless of their geographic location or financial "
        "circumstances. Early childhood education has emerged as a particularly high-impact "
        "investment, with research consistently showing that quality preschool programs yield "
        "long-term benefits in cognitive development and academic achievement. Teacher quality "
        "remains the single most important factor in student learning, underscoring the need to "
        "attract, train, and retain talented educators."
    ),
)

# ---------------------------------------------------------------------------
# English — Male (am_adam for 1-3, am_michael for 4-6)
# ---------------------------------------------------------------------------

_EN_M_001 = AudioScript(
    script_id="en_m_001_technology",
    filename="en_male_001_technology.wav",
    language="en",
    gender="male",
    voice="am_adam",
    topic="technology",
    expected_keywords=["technology", "artificial intelligence", "digital transformation", "innovation", "software"],
    text=(
        "Technology is transforming every aspect of modern life at an unprecedented pace, reshaping "
        "how we work, communicate, learn, and interact with the world around us. Artificial "
        "intelligence, perhaps the most consequential technological development of our era, is "
        "advancing rapidly, with applications ranging from medical diagnosis and drug discovery to "
        "autonomous vehicles and creative content generation. The smartphone has become an "
        "indispensable tool for billions of people worldwide, putting the sum of human knowledge "
        "in our pockets while simultaneously raising concerns about privacy, addiction, and social "
        "isolation. Cloud computing has fundamentally changed how businesses store and process "
        "data, enabling even small companies to access computing power once available only to large "
        "corporations. Cybersecurity has emerged as a critical concern as increasingly important "
        "systems — from power grids to financial markets — depend on complex interconnected digital "
        "infrastructure vulnerable to attack."
    ),
)

_EN_M_002 = AudioScript(
    script_id="en_m_002_sports",
    filename="en_male_002_sports.wav",
    language="en",
    gender="male",
    voice="am_adam",
    topic="sports",
    expected_keywords=["sports", "competition", "athletes", "Olympic Games", "performance"],
    text=(
        "Sports occupy a unique place in human culture, combining physical achievement, competition, "
        "and entertainment in ways that transcend language, culture, and geography. The Olympic "
        "Games, held every four years, represent the pinnacle of international sporting competition, "
        "bringing together athletes from over two hundred countries to compete in a spirit of mutual "
        "respect and peaceful rivalry. Professional sports leagues have become billion-dollar "
        "industries, with franchises, broadcasting rights, and player salaries reaching "
        "extraordinary levels. The science of sports performance has advanced dramatically in recent "
        "decades, with sophisticated training methods, nutrition science, and data analytics helping "
        "athletes push the boundaries of human physical achievement. Youth sports play an important "
        "developmental role, teaching children teamwork, discipline, resilience, and the ability to "
        "manage both victory and defeat — skills that translate well beyond the playing field."
    ),
)

_EN_M_003 = AudioScript(
    script_id="en_m_003_history",
    filename="en_male_003_history.wav",
    language="en",
    gender="male",
    voice="am_adam",
    topic="world_history",
    expected_keywords=["history", "civilization", "ancient Rome", "Industrial Revolution", "World War"],
    text=(
        "History is the study of the human past, offering essential context for understanding the "
        "present and navigating the future. The development of writing systems in ancient "
        "Mesopotamia and Egypt around five thousand years ago marked a turning point in human "
        "civilization, enabling the accumulation and transmission of knowledge across generations. "
        "The Renaissance, beginning in fourteenth-century Italy, witnessed a remarkable flowering "
        "of art, science, and philosophy that laid the groundwork for the modern Western world. "
        "The Industrial Revolution, which began in Britain in the late eighteenth century, "
        "transformed human societies more rapidly and profoundly than any previous development, "
        "creating urban industrial economies and new social classes. The twentieth century was "
        "marked by two devastating world wars, the rise and fall of totalitarian regimes, the "
        "development of nuclear weapons, and the gradual decolonization of Asia and Africa."
    ),
)

_EN_M_004 = AudioScript(
    script_id="en_m_004_economics",
    filename="en_male_004_economics.wav",
    language="en",
    gender="male",
    voice="am_michael",
    topic="economics",
    expected_keywords=["economics", "GDP", "trade", "inequality", "markets"],
    text=(
        "Economics is the study of how societies allocate scarce resources to satisfy unlimited "
        "human wants and needs. The discipline has evolved considerably since Adam Smith published "
        "The Wealth of Nations in 1776, branching into macroeconomics, which examines the behavior "
        "of entire economies, and microeconomics, which focuses on individual markets and actors. "
        "Global trade has expanded dramatically over the past century, connecting national economies "
        "through complex supply chains and creating both enormous wealth and significant dislocations "
        "in labor markets. Income inequality has risen in many countries, prompting debate about "
        "the proper role of government in redistributing wealth and ensuring that the gains from "
        "economic growth are broadly shared. Behavioral economics has challenged the classical "
        "assumption of perfectly rational economic actors, showing that human decision-making is "
        "often influenced by cognitive biases and emotional factors."
    ),
)

_EN_M_005 = AudioScript(
    script_id="en_m_005_space",
    filename="en_male_005_space.wav",
    language="en",
    gender="male",
    voice="am_michael",
    topic="space_exploration",
    expected_keywords=["space exploration", "NASA", "Mars", "astronauts", "satellites"],
    text=(
        "Space exploration represents one of humanity's most ambitious undertakings, driven by "
        "curiosity, scientific inquiry, national prestige, and the long-term imperative of becoming "
        "a multi-planetary species. The Space Age began in 1957 with the Soviet Union's launch of "
        "Sputnik, the first artificial satellite, followed by Yuri Gagarin's historic first human "
        "spaceflight in 1961 and Neil Armstrong's moonwalk in 1969. After decades dominated by "
        "government agencies like NASA and Roscosmos, private companies led by visionary "
        "entrepreneurs have revolutionized the industry, dramatically reducing launch costs and "
        "accelerating the pace of innovation. Mars has become the next major target for human "
        "exploration, with both government and private missions working toward establishing a "
        "permanent human presence on the Red Planet. The search for extraterrestrial life, whether "
        "microbial organisms on Mars or signals from distant civilizations, remains one of the most "
        "profound scientific questions of our time."
    ),
)

_EN_M_006 = AudioScript(
    script_id="en_m_006_philosophy",
    filename="en_male_006_philosophy.wav",
    language="en",
    gender="male",
    voice="am_michael",
    topic="philosophy",
    expected_keywords=["philosophy", "ethics", "Socrates", "reasoning", "wisdom"],
    text=(
        "Philosophy, the love of wisdom, is the foundational discipline from which most other "
        "fields of human inquiry have emerged over the course of Western intellectual history. "
        "Ancient Greek philosophers including Socrates, Plato, and Aristotle asked fundamental "
        "questions about reality, knowledge, ethics, and politics that continue to resonate and "
        "challenge us today. The rationalist and empiricist traditions of early modern philosophy, "
        "represented by thinkers like Descartes, Locke, Hume, and Kant, grappled with the nature "
        "of human knowledge and the limits of reason. Contemporary philosophy addresses a wide "
        "range of pressing issues, from the ethics of artificial intelligence and genetic "
        "engineering to questions of justice, democracy, and the meaning of life in a secular age. "
        "Philosophy teaches rigorous argumentation, conceptual clarity, and the ability to examine "
        "assumptions — skills of lasting value in any intellectual or professional domain."
    ),
)

# ---------------------------------------------------------------------------
# Master list (25 scripts total)
# ---------------------------------------------------------------------------

_ALL_SCRIPTS: list[AudioScript] = [
    # Italian Female (6)
    _IT_F_001, _IT_F_002, _IT_F_003, _IT_F_004, _IT_F_005, _IT_F_006,
    # Italian Male (7)
    _IT_M_001, _IT_M_002, _IT_M_003, _IT_M_004, _IT_M_005, _IT_M_006, _IT_M_007,
    # English Female (6)
    _EN_F_001, _EN_F_002, _EN_F_003, _EN_F_004, _EN_F_005, _EN_F_006,
    # English Male (6)
    _EN_M_001, _EN_M_002, _EN_M_003, _EN_M_004, _EN_M_005, _EN_M_006,
]


class ScriptBuilder:
    """Provides access to the static dataset script definitions."""

    def get_all_scripts(self) -> list[AudioScript]:
        return list(_ALL_SCRIPTS)

    def get_scripts_by_language(self, lang: str) -> list[AudioScript]:
        return [s for s in _ALL_SCRIPTS if s.language == lang]

    def get_scripts_by_topic(self, topic: str) -> list[AudioScript]:
        return [s for s in _ALL_SCRIPTS if s.topic == topic]
