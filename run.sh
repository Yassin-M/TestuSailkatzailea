#!/bin/bash

# JAR fitxategiaren path-a
JAR_PATH="TestuSailkatzailea.jar"

# JAR fitxategia existitzen den egiaztatu
if [ ! -f "$JAR_PATH" ]; then
    echo "Errorea: Ezin da $JAR_PATH aurkitu. Ziurtatu JAR fitxategia dagoen direktorioan exekutatzen ari zarela."
    exit 1
fi

echo "=== 1. CSV -> ARFF ==="
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" csv2arff data/tweetSentiment.train.csv
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" csv2arff data/tweetSentiment.dev.csv
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" csv2arff data/tweetSentiment.test_blind.csv

echo "=== 2. Aurreprozesamendua ==="
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" aurreprozesatu data/arff/raw/tweetSentiment.train.arff
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" aurreprozesatu data/arff/raw/tweetSentiment.dev.arff
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" aurreprozesatu data/arff/raw/tweetSentiment.test_blind.arff

echo "=== 3. Bektorizazio mota optimoa aztertu ==="
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" bektorizazioOptimoa data/arff/raw/tweetSentiment.train.arff data/arff/raw/tweetSentiment.dev.arff

echo "=== 4. Datuak bektorizatu ==="
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" datuakBektorizatu data/arff/raw/tweetSentiment.train.arff data/arff/raw/tweetSentiment.dev.arff data/arff/raw/tweetSentiment.test_blind.arff

echo "=== 5. Parametro ekorketa ==="
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" parametroEkorketa data/arff/bektorizatuta/train_bektorizatua.arff

echo "=== 6. Kalitate txostena ==="
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" kalitateTxostena data/arff/bektorizatuta/train_bektorizatua.arff data/arff/bektorizatuta/dev_bektorizatua.arff

echo "=== 7. Amaierako eredua sortu ==="
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" sailkatzaileaSortu data/eredua/bestBayesNetConfig.txt data/arff/bektorizatuta/train_bektorizatua.arff data/arff/bektorizatuta/dev_bektorizatua.arff

echo "=== 8. Iragarpenak egin ==="
java --add-opens java.base/java.lang=ALL-UNNAMED -jar "$JAR_PATH" iragarpenakEgin data/tweetSentiment.test_blind.csv

echo ">>> Pipeline osoa amaituta. <<<"
