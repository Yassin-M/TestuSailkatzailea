# Testu Sailkatzailea

Weka API-a erabiliz garatutako testu-sailkatzailea
Proiektu honek sailkapen-prozesuaren pipeline guztia kudeatzen du: hasierako datuen garbiketa eta bihurketatik hasi, test blind baten aurrean iragarpenak egiteko `BayesNet` eredu optimizatu bat sortu arte.

---

## For the impatients

Jatorrizko CSV datuak `data/` karpetan badaude eta `TestuSailkatzailea.jar` aplikazioa proiektuaren karpetan badago, exekutatu zuzenean bash script-a pauso guztiak banan-banan eta automatikoki garatzeko:

```bash
chmod +x run.sh
./run.sh
```
*Script honek ziklo osoa egingo du pauso bat bestearen atzetik: CSV-k ARFF bihurtu, aurreprozesatu, bektorizazio optimoa aztertu, bektorizatu, parametro ekorketa egin, kalitate-txostena sortu, eredu finala sortu Train eta Dev elkartuz, eta azkenik, test-blind datuen iragarpenak egin.*

**Java bertsioei buruzko oharra:**
Baliteke arazoak izatea Java bertsio batzuekin. Script-aren barruan edo eskuzko exekuzioan `--add-opens java.base/java.lang=ALL-UNNAMED` kendu daiteke.

## Exekuzio Moduak eta Komandoak

`.jar` fitxategiak komando nagusi bat onartzen du lehen argumentu gisa, eta ondotik beste parametro batzuk jaso ditzake. Parametro gehienak **hautazkoak** dira; zehazten ez badira, programak kodean bertan definitutako berezko kokalekuak (defektuzekoak) erabiliko ditu.

Erabilera orokorra:
```bash
java -jar TestuSailkatzailea.jar <KOMANDOA> [PARAMETROAK]
```

Erabilgarri dauden komando guztiak eta beren xehetasunak:

### 1. `csv2arff`
CSV formatuan dagoen datu-fitxategi bat Weka-ren jatorrizko ARFF formatura bihurtzen du.
* **Erabilera:** `java -jar TestuSailkatzailea.jar csv2arff <csv_path>`
* **Parametroak:**
  * `<csv_path>` *(Derrigorrezkoa)*: Jatorrizko CSV fitxategiaren path-a.

### 2. `aurreprozesatu`
Aurreprozesamendua eta garbiketa aplikatzen dizkie prozesatu gabeko (raw) ARFF fitxategiei. Txioetako testua normalizatzen du.
* **Erabilera:** `java -jar TestuSailkatzailea.jar aurreprozesatu <arff_path>`
* **Parametroak:**
  * `<arff_path>` *(Derrigorrezkoa)*: Garbitu nahi den ARFF fitxategiaren path-a.

### 3. `bektorizazioOptimoa`
Testu hutsa (String) atributu neurgarrien bektore bihurtzeko zein den konfigurazio edo dimentsionalitate onena aztertuko du.
* **Erabilera:** `java -jar TestuSailkatzailea.jar bektorizazioOptimoa [train_arff] [test_arff]`
* **Parametroak:**
  * `[train_arff]` *(Hautazkoa)*: Entrenamenduko ARFF fitxategia (garbituta). _Defektuz: `data/arff/raw/tweetSentiment.train.arff`_
  * `[test_arff]` *(Hautazkoa)*: Dev/Test ARFF fitxategia. _Defektuz: `data/arff/raw/tweetSentiment.dev.arff`_

### 4. `datuakBektorizatu`
Aurrez zehaztutako bektorizazio onena aplikatzen die datu-multzoei, testua entrenamendurako erabilgarriak diren balioetara aldatuz eta diskoan gordez.
* **Erabilera:** `java -jar TestuSailkatzailea.jar datuakBektorizatu [train_arff] [dev_arff] [test_blind_arff]`
* **Parametroak:**
  * `[train_arff]` *(Hautazkoa)*: _Defektuz: `data/arff/raw/tweetSentiment.train.arff`_
  * `[dev_arff]` *(Hautazkoa)*: _Defektuz: `data/arff/raw/tweetSentiment.dev.arff`_
  * `[test_blind_arff]` *(Hautazkoa)*: _Defektuz: `data/arff/raw/tweetSentiment.test_blind.arff`_

### 5. `parametroEkorketa`
BayesNet sailkatzailearen parametro optimoenak topatzeko *Fine-Tuning* prozesua exekutatzen du.
* **Erabilera:** `java -jar TestuSailkatzailea.jar parametroEkorketa [train_bektorizatua_arff]`
* **Parametroak:**
  * `[train_bektorizatua_arff]` *(Hautazkoa)*: Bektorizatutako train datuak. _Defektuz: `data/arff/bektorizatuta/train_bektorizatua.arff`_

### 6. `kalitateTxostena`
Parametro optimoak dituen eredua ebaluatzen du (Test/Dev datuekin) eta kalitate-txostena sortzen du metrika nagusiekin (F-Measure, Precision, Recall...).
* **Erabilera:** `java -jar TestuSailkatzailea.jar kalitateTxostena [train_bektorizatua_arff] [test_bektorizatua_arff]`
* **Parametroak:**
  * `[train_bektorizatua_arff]` *(Hautazkoa)*: Bektorizatutako train fitxategia. _Defektuz: `data/arff/bektorizatuta/train_bektorizatua.arff`_
  * `[test_bektorizatua_arff]` *(Hautazkoa)*: Bektorizatutako dev/test fitxategia. _Defektuz: `data/arff/bektorizatuta/dev_bektorizatua.arff`_

### 7. `sailkatzaileaSortu`
Amaierako eredu definitiboa edo `.model` fitxategia eraikitzen du. Horretarako, Train eta Dev fitxategiak batzen ditu datu bolumena handitzeko.
* **Erabilera:** `java -jar TestuSailkatzailea.jar sailkatzaileaSortu [config_txt] [train_bektorizatua_arff] [test_bektorizatua_arff]`
* **Parametroak:**
  * `[config_txt]` *(Hautazkoa)*: Ereduaren konfigurazio fitxategia. _Defektuz: `data/eredua/bestBayesNetConfig.txt`_
  * `[train_bektorizatua_arff]` *(Hautazkoa)*: Bektorizatutako train fitxategia. _Defektuz: `data/arff/bektorizatuta/train_bektorizatua.arff`_
  * `[test_bektorizatua_arff]` *(Hautazkoa)*: Bektorizatutako dev fitxategia. _Defektuz: `data/arff/bektorizatuta/dev_bektorizatua.arff`_

### 8. `iragarpenakEgin`
Aurreko pausoan gordetako eredu finala kargatzen du, eta klase ezezaguna duten datuen iragarpenak egiten ditu (output fitxategi batean idatziz).
* **Erabilera:** `java -jar TestuSailkatzailea.jar iragarpenakEgin [test_blind_csv]`
* **Parametroak:**
  * `[test_blind_csv]` *(Hautazkoa)*: Iragarri nahi den CSV fitxategia. _Defektuz: `data/tweetSentiment.test_blind.csv`_

### 9. `pipelineOsoa`
1etik 8ra dauden urrats guztiak modu sekuentzial eta jarraituan pasatzen ditu.
* **Erabilera:** `java -jar TestuSailkatzailea.jar pipelineOsoa [train_csv] [dev_csv] [test_blind_csv]`
* **Parametroak:**
  * `[train_csv]` *(Hautazkoa)*: _Defektuz: `data/tweetSentiment.train.csv`_
  * `[dev_csv]` *(Hautazkoa)*: _Defektuz: `data/tweetSentiment.dev.csv`_
  * `[test_blind_csv]` *(Hautazkoa)*: _Defektuz: `data/tweetSentiment.test_blind.csv`_

---
