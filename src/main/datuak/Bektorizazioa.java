package main.datuak;

import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;
import weka.core.SerializationHelper;
import weka.core.Utils;
import weka.core.converters.ConverterUtils;
import weka.core.stemmers.IteratedLovinsStemmer;
import weka.core.stemmers.NullStemmer;
import weka.core.tokenizers.AlphabeticTokenizer;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Random;
/**
 * Testu-datuak bektore numeriko bihurtzeaz (bektorizazioa) arduratzen den klasea.
 * Tresna honek aurre-prozesamendu konfigurazio optimoa (TF-IDF, Stemming, Tokenizazioa)
 * bilatzeko eta informazio-irabazian (InfoGain) oinarritutako atributu-hautapena
 * aplikatzeko aukera ematen du.
 */
public class Bektorizazioa {
    /** Testua bektore bihurtzeko iragazki nagusia (entrenamenduan erabilia). */
    private static StringToWordVector stwv;
    /** Hiztegi finkoa erabiltzen duen iragazkia, test multzoek entrenamenduko hitz berdinak erabil ditzaten.*/
    private static FixedDictionaryStringToWordVector fdstwv;
    /** Atributu garrantzitsuenak hautatzeko iragazkia (InfoGain bidez). */
    private static AttributeSelection as;

    /**
     * Bektorizazio konfigurazio hoberena aztertzen eta ebaluatzen du garapen-multzo bat erabiliz.
     * Hiztegia gordetzen du eta Naive Bayes sailkatzailea aplikatzen du emaitzak lortzeko.
     *
     * @param data Entrenamendurako datu-multzoa.
     * @param test Konfigurazioa balioztatzeko garapen-multzoa (Dev).
     * @param hiztegia Mantendu beharreko hitz kopuru maximoa (hiztegiaren tamaina).
     * @throws Exception Iragazkiak aplikatzean edo ebaluazioan errorerik gertatuz gero.
     */
    public static void bektorizazioMotaEgokienaAztertu(Instances data, Instances test, int hiztegia) throws Exception{
        konfigurazioEgokienaAukeratu(data, hiztegia);

        if(data.classIndex() == -1) data.setClassIndex(data.attribute("Sentiment").index());
        if(test.classIndex() == -1) test.setClassIndex(test.attribute("Sentiment").index());

        Files.createDirectories(Paths.get("data/arff/bektorizatuta/txt"));
        stwv.setDictionaryFileToSaveTo(new File("data/arff/bektorizatuta/txt/bestDictionary.txt"));

        weka.filters.MultiFilter multiFilter = new weka.filters.MultiFilter();
        multiFilter.setFilters(new weka.filters.Filter[] { stwv, as });

        weka.classifiers.meta.FilteredClassifier fc = new weka.classifiers.meta.FilteredClassifier();
        fc.setFilter(multiFilter);
        fc.setClassifier(new NaiveBayes());

        fc.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        eval.evaluateModel(fc, test);

        System.out.println("DEV multzoarekin ebaluatu ostean:");
        System.out.println("Accuracy: " + eval.pctCorrect() + "%");
    }

    /**
     * Train, dev eta test_blind itsuaren bektorizazio definitiboa egiten du.
     *
     * @param train Jatorrizko entrenamendu-multzoa.
     * @param dev Jatorrizko garapen-multzoa (Dev).
     * @param testBlind Jatorrizko test itsua (etiketarik gabea).
     * @throws Exception Fitxategiak irakurtzean edo iragazkiak aplikatzean errorerik gertatuz gero.
     */
    public static void datuakBektorizatu(Instances train, Instances dev, Instances testBlind) throws Exception {
        String configTxt = new String(Files.readAllBytes(Paths.get("data/arff/bektorizatuta/txt/bektorizazioHoberena.txt")));
        stwv = new StringToWordVector();
        stwv.setOptions(weka.core.Utils.splitOptions(configTxt));

        stwv.setDictionaryFileToSaveTo(new File("data/arff/bektorizatuta/txt/bestDictionary.txt"));

        as = (AttributeSelection) SerializationHelper.read("data/arff/bektorizatuta/filter/bestAttributeSelection.filter");

        if(train.classIndex() == -1) train.setClassIndex(train.attribute("Sentiment").index());
        if(dev.classIndex() == -1) dev.setClassIndex(dev.attribute("Sentiment").index());
        if(testBlind.classIndex() == -1) testBlind.setClassIndex(testBlind.attribute("Sentiment").index());

        stwv.setInputFormat(train);
        Instances trainBek = Filter.useFilter(train, stwv);

        setFdstwv();
        fdstwv.setDictionaryFile(new File("data/arff/bektorizatuta/txt/bestDictionary.txt"));

        fdstwv.setInputFormat(dev);
        Instances devBek = Filter.useFilter(dev, fdstwv);

        fdstwv.setInputFormat(testBlind);
        Instances testBek = Filter.useFilter(testBlind, fdstwv);

        Instances trainFinal = Filter.useFilter(trainBek, as);
        Instances devFinal = Filter.useFilter(devBek, as);
        Instances testFinal = Filter.useFilter(testBek, as);

        Files.createDirectories(Paths.get("data/arff/bektorizatuta"));
        ConverterUtils.DataSink.write("data/arff/bektorizatuta/train_bektorizatua.arff", trainFinal);
        ConverterUtils.DataSink.write("data/arff/bektorizatuta/dev_bektorizatua.arff", devFinal);
        ConverterUtils.DataSink.write("data/arff/bektorizatuta/test_blind_bektorizatua.arff", testFinal);

        System.out.println("Train, Dev eta TestBlind arrakastaz bektorizatu eta gorde dira 'data/arff/bektorizatuta/' karpetan.");
    }

    /**
     * Konfigurazio hoberena bilatzeko bilaketa sakona egiten du konbinazio hauen artean:
     * - Bektorizazio-espazio mota (Bitarra, TF, TF-IDF).
     * - Stemmer-a (IteratedLovins edo batere ez).
     * - Tokenizatzailea (Alphabetikoa edo NGram 1-2).
     * Ebaluazioa 10-fold cross-validation bidez egiten da Naive Bayes erabiliz.
     *
     * @param train Bilaketarako erabiliko den entrenamendu-multzoa.
     * @param hiztegia Hautatu beharreko atributu kopurua.
     * @throws Exception Ebaluazio eskeman errorerik gertatuz gero.
     */
    private static void konfigurazioEgokienaAukeratu(Instances train, int hiztegia) throws Exception{
        if(train.classIndex()==-1) train.setClassIndex(train.attribute("Sentiment").index());
        // probatu nahi ditugun aukerak
        boolean bestStem = false;
        // int bestN = 1;
        double bestAccuracy = -1.0;
        int bestMota = -1;
        int bestTok = -1;
        int i = 1;
        for(int mota = 0; mota<3; mota++){
            for(boolean stemmer: new boolean[]{false, true}){// stemmer erabili ala ez
                for(int tokenizer=0; tokenizer<=1; tokenizer++){
                    StringToWordVector unekoFiltroa = new StringToWordVector();
                    unekoFiltroa.setWordsToKeep(hiztegia);
                    unekoFiltroa.setLowerCaseTokens(true);
                    unekoFiltroa.setAttributeNamePrefix("W_");

                    filtroaPrestatu(unekoFiltroa, mota);
                    stemmerEzarri(unekoFiltroa, stemmer);
                    tokenizerEzarri(unekoFiltroa, tokenizer);

                    unekoFiltroa.setInputFormat(train);
                    Instances bektorizatuak = Filter.useFilter(train, unekoFiltroa);

                    AttributeSelection unekoAS = new AttributeSelection();
                    InfoGainAttributeEval eval = new InfoGainAttributeEval();
                    Ranker ranker = new Ranker();
                    ranker.setNumToSelect(hiztegia);

                    unekoAS.setEvaluator(eval);
                    unekoAS.setSearch(ranker);
                    unekoAS.setInputFormat(bektorizatuak);

                    Instances finala = Filter.useFilter(bektorizatuak, unekoAS);

                    Evaluation ev = new Evaluation(finala);
                    ev.crossValidateModel(new NaiveBayes(), finala, 10, new Random(i));

                    double acc = ev.pctCorrect();
                    System.out.println("Mota: " + getMotaIzena(mota) + " | Stem: " + getStemmerIzena(stemmer) + " | Tok: " + getTokenizerIzena(tokenizer) + " -> Acc: " + acc);

                    if (acc > bestAccuracy) {
                        bestAccuracy = acc;
                        bestMota = mota;
                        bestStem = stemmer;
                        bestTok = tokenizer;
                        setStwv(unekoFiltroa);
                        as = unekoAS;
                    }
                    i++;
                }
            }
        }
        System.out.println("Hoberena --> Mota: " + getMotaIzena(bestMota) + " | Stem: " + getStemmerIzena(bestStem) + " | Tok: " + getTokenizerIzena(bestTok) + " -> Acc: " + bestAccuracy);

        Files.createDirectories(Paths.get("data/arff/bektorizatuta/filter"));
        Files.createDirectories(Paths.get("data/arff/bektorizatuta/txt"));
        SerializationHelper.write("data/arff/bektorizatuta/filter/bestAttributeSelection.filter", as);
        String[] config = stwv.getOptions();
        String configTxt = Utils.joinOptions(config);
        Files.write(Paths.get("data/arff/bektorizatuta/txt/bektorizazioHoberena.txt"), configTxt.getBytes());
    }

    /**
     * FixedDictionaryStringToWordVector objektua konfiguratzen du hautatutako
     * StringToWordVector iragazkiaren parametro berdinekin.
     */
    private static void setFdstwv(){
        StringToWordVector stringToWordVector = getStwv();
        fdstwv = new FixedDictionaryStringToWordVector();
        fdstwv.setTFTransform(stringToWordVector.getTFTransform());
        fdstwv.setIDFTransform(stringToWordVector.getIDFTransform());
        fdstwv.setOutputWordCounts(stringToWordVector.getOutputWordCounts());

        if (stringToWordVector.getDictionaryFileToSaveTo() != null) {
            fdstwv.setDictionaryFile(stringToWordVector.getDictionaryFileToSaveTo());
        }

        fdstwv.setTokenizer(stringToWordVector.getTokenizer());
        fdstwv.setStemmer(stringToWordVector.getStemmer());
    }

    private static StringToWordVector getStwv(){ return stwv; }
    private static void setStwv(StringToWordVector pStwv){ stwv = pStwv; }

    /**
     * Filtoraren bektorizazio-mota konfiguratzen du (Bitarra, TF edo TF-IDF).
     * @param f Konfiguratuko den iragazkia.
     * @param mota Moduaren indizea (0: Bitarra, 1: TF, 2: TF-IDF).
     */
    private static void filtroaPrestatu(StringToWordVector f, int mota){
        if(mota==0){
            f.setOutputWordCounts(false);
            f.setTFTransform(false);
            f.setIDFTransform(false);
        }else if(mota == 1){
            f.setOutputWordCounts(true);
            f.setTFTransform(true);
            f.setIDFTransform(false);
        }else if(mota == 2){
            f.setOutputWordCounts(true);
            f.setTFTransform(true);
            f.setIDFTransform(true);
        }
    }

    /**
     * Filtroari stemmer-a ezartzen dio.
     * @param f Iragazkia.
     * @param stem True IteratedLovins erabili nahi bada, false NullStemmer-erako.
     */
    private static void stemmerEzarri(StringToWordVector f, boolean stem){
        if(stem){
            f.setStemmer(new IteratedLovinsStemmer());
        }else{
            f.setStemmer(new NullStemmer());
        }
    }

    /**
     * Filtroari tokenizatzaile mota ezartzen dio.
     * @param f Filtroa.
     * @param tokenizer 0 AlphabeticTokenizer-erako, beste edozein NGram(1-2)-rako.
     */
    private static void tokenizerEzarri(StringToWordVector f, int tokenizer){
        if(tokenizer==0){
            f.setTokenizer(new AlphabeticTokenizer());
        }else{
            NGramTokenizer ngram = new NGramTokenizer();
            ngram.setNGramMinSize(1);
            ngram.setNGramMaxSize(2);
            f.setTokenizer(ngram);
        }
    }
    /** Bektorizazio-espazioaren izen lagungarria itzultzen du. */
    private static String getMotaIzena(int mota) {
        return switch (mota) {
            case 0 -> "Bitarra";
            case 1 -> "TF";
            case 2 -> "TF-IDF";
            default -> "Ezezaguna";
        };
    }

    /** Stemmer-aren izen lagungarria itzultzen du. */
    private static String getStemmerIzena(boolean stem) { return stem ? "IteratedLovins" : "None"; }

    /** Tokenizatzailearen izen lagungarria itzultzen du. */
    private static String getTokenizerIzena(int tok) { return (tok == 0) ? "Alphabetic" : "NGram(1-2)"; }
}