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

public class Bektorizazioa {
    private static StringToWordVector stwv;
    private static FixedDictionaryStringToWordVector fdstwv;
    private static AttributeSelection as;

    public static void bektorizazioMotaEgokienaAztertu(Instances data, Instances test) throws Exception{
        konfigurazioEgokienaAukeratu(data);

        if(data.classIndex() == -1) data.setClassIndex(data.attribute("Sentiment").index());
        if(test.classIndex() == -1) test.setClassIndex(test.attribute("Sentiment").index());

        Files.createDirectories(Paths.get("dataFinala/txt"));
        stwv.setDictionaryFileToSaveTo(new File("dataFinala/txt/bestDictionary.txt"));

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

    public static void datuakBektorizatu(Instances train, Instances dev, Instances testBlind) throws Exception {
        String configTxt = new String(Files.readAllBytes(Paths.get("dataFinala/txt/bektorizazioHoberena.txt")));
        stwv = new StringToWordVector();
        stwv.setOptions(weka.core.Utils.splitOptions(configTxt));

        stwv.setDictionaryFileToSaveTo(new File("dataFinala/txt/bestDictionary.txt"));

        as = (AttributeSelection) SerializationHelper.read("dataFinala/filter/bestAttributeSelection.filter");

        if(train.classIndex() == -1) train.setClassIndex(train.attribute("Sentiment").index());
        if(dev.classIndex() == -1) dev.setClassIndex(dev.attribute("Sentiment").index());
        if(testBlind.classIndex() == -1) testBlind.setClassIndex(testBlind.attribute("Sentiment").index());

        stwv.setInputFormat(train);
        Instances trainBek = Filter.useFilter(train, stwv);

        setFdstwv();
        fdstwv.setDictionaryFile(new File("dataFinala/txt/bestDictionary.txt"));

        fdstwv.setInputFormat(dev);
        Instances devBek = Filter.useFilter(dev, fdstwv);

        fdstwv.setInputFormat(testBlind);
        Instances testBek = Filter.useFilter(testBlind, fdstwv);

        Instances trainFinal = Filter.useFilter(trainBek, as);
        Instances devFinal = Filter.useFilter(devBek, as);
        Instances testFinal = Filter.useFilter(testBek, as);

        Files.createDirectories(Paths.get("dataFinala/arff"));
        ConverterUtils.DataSink.write("dataFinala/arff/train_bektorizatua.arff", trainFinal);
        ConverterUtils.DataSink.write("dataFinala/arff/dev_bektorizatua.arff", devFinal);
        ConverterUtils.DataSink.write("dataFinala/arff/test_blind_bektorizatua.arff", testFinal);

        System.out.println("Train, Dev eta TestBlind arrakastaz bektorizatu eta gorde dira 'dataFinala/arff/' karpetan.");
    }

    private static void konfigurazioEgokienaAukeratu(Instances train) throws Exception{
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
                    unekoFiltroa.setWordsToKeep(1000);
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
                    ranker.setNumToSelect(1000);

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

        Files.createDirectories(Paths.get("dataFinala/filter"));
        Files.createDirectories(Paths.get("dataFinala/txt"));
        SerializationHelper.write("dataFinala/filter/bestAttributeSelection.filter", as);
        String[] config = stwv.getOptions();
        String configTxt = Utils.joinOptions(config);
        Files.write(Paths.get("dataFinala/txt/bektorizazioHoberena.txt"), configTxt.getBytes());
    }

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

    private static void stemmerEzarri(StringToWordVector f, boolean stem){
        if(stem){
            f.setStemmer(new IteratedLovinsStemmer());
        }else{
            f.setStemmer(new NullStemmer());
        }
    }

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

    private static String getMotaIzena(int mota) {
        return switch (mota) {
            case 0 -> "Bitarra";
            case 1 -> "TF";
            case 2 -> "TF-IDF";
            default -> "Ezezaguna";
        };
    }

    private static String getStemmerIzena(boolean stem) { return stem ? "IteratedLovins" : "None"; }
    private static String getTokenizerIzena(int tok) { return (tok == 0) ? "Alphabetic" : "NGram(1-2)"; }
}