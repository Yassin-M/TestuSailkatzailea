package main;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;

import java.io.*;
import java.nio.charset.StandardCharsets;
/**
 * CSV fitxategiak irakurri, formatu-erroreak garbitu eta Weka-k onartzen duen
 * ARFF formatura bihurtzeaz arduratzen den klasea.
 * <p>
 *     Bereiziki diseinatuta dago testu eremuekin (Tweet-ak) arazoak ematen dituzten
 *     komatxo bikoitzak eta kontrol-karaktereak kudeatzeko.
 * </p>
 */
public class CSV2Arff {
    /**
     * CSV batetik ARFF-rako bihurketa prozesu osoa kudeatzen du.
     * Lehenik garbiketa fisikoa egiten du eta ondoren Wekaren CSVLoader-a erabiltzen du.
     *
     * @param arg Prozesatu nahi den CSV fitxategiaren bidea (path).
     * @throws Exception Weka-rekin, fitxategien sarbidearekin edo formatuarekin arazoren bat egonez gero.
     */
    public static void arffPasatu(String arg) throws Exception{
        String inputPath = arg;
        String cleanedPath = "data/clean/sortaGarbia.csv";
        String outputPath = "data/arff/sortaGarbia.arff";
        cleanCSV(inputPath, cleanedPath);

        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(cleanedPath));
        loader.setNominalAttributes("1,2");
        loader.setStringAttributes("3,4,5");
        Instances data = loader.getDataSet();

        ArffSaver saver = new ArffSaver();
        saver.setInstances(data);
        saver.setFile(new File(outputPath));
        saver.writeBatch();
    }
    /**
     * CSV fitxategi bat lerroz lerro irakurtzen du eta erroreak sortzen dituzten
     * erregistroak iragazten ditu (komatxo desorekatuak, zutabe kopuru okerra, etab.).
     * @param inputPath Jatorrizko CSV fitxategiaren bidea.
     * @param cleanPath Sortuko den CSV fitxategi garbi eta normalizatuaren bidea.
     */
    public static void cleanCSV(String inputPath, String cleanPath){
        int esperotutakoZutabeak = 0;
        int lerroEzabatuak = 0;
        int lerroTotalak = 0;

        try(BufferedReader br = new BufferedReader(new FileReader(inputPath, StandardCharsets.UTF_8));
            BufferedWriter bw = new BufferedWriter(new FileWriter(cleanPath, StandardCharsets.UTF_8))){
            String unekoLerroa;
            boolean lehenLerroa = true;
            while((unekoLerroa = br.readLine())!=null){
                lerroTotalak++;
                String trimmedLerroa = unekoLerroa.trim();

                if(trimmedLerroa.isEmpty()) continue;

                if(!hasBalancedQuotes(trimmedLerroa)){
                    lerroEzabatuak++;
                    continue;
                }

                if(!trimmedLerroa.endsWith("\"")){
                    lerroEzabatuak++;
                    continue;
                }

                String[] zutabeak = trimmedLerroa.split(",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", -1);
                if(lehenLerroa){
                    esperotutakoZutabeak = zutabeak.length;
                    bw.write(lerroaBireraiki(zutabeak));
                    bw.newLine();
                    lehenLerroa = false;
                }else{
                    if(zutabeak.length==esperotutakoZutabeak){
                        bw.write(lerroaBireraiki(zutabeak));
                        bw.newLine();
                    }else{
                        lerroEzabatuak++;
                    }
                }
            }
            System.out.println("Prozesatutako lerroak: " + lerroTotalak);
            System.out.println("Ezabatutako lerroak: " + lerroEzabatuak);
        }catch(IOException e){
            e.printStackTrace();
        }
    }

    /**
     * CSV erregistro bat bireraikitzen du, gelaxka bakoitzeko komatxoak normalizatuz.
     * Komatxo bikoitz guztiak sinpleetan (') bihurtzen ditu eta kontrol-karaktereak ezabatzen ditu.
     * @param zutabeak Erregistroaren atributuak biltzen dituen array-a.
     */
    private static String lerroaBireraiki(String[] zutabeak) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < zutabeak.length; i++) {
            String gelaxka = zutabeak[i].trim();
            gelaxka = gelaxka.replace("“", "'")
                    .replace("”", "'")
                    .replace("‘", "'")
                    .replace("’", "'");

            gelaxka = gelaxka.replaceAll("[\\p{Cntrl}&&[^\r\n\t]]", "");

            if (gelaxka.startsWith("\"") && gelaxka.endsWith("\"") && gelaxka.length() >= 2) {
                gelaxka = gelaxka.substring(1, gelaxka.length() - 1);
            }

            gelaxka = gelaxka.replace("\"", "'");

            sb.append("\"").append(gelaxka).append("\"");

            if (i < zutabeak.length - 1) {
                sb.append(",");
            }
        }
        return sb.toString();
    }

    /**
     * Lerro batean komatxo bikoitz kopurua bikoitia dela egiaztatzen du.
     * Hau ezinbestekoa da CSV formatuan erregistro bat moztuta ez dagoela jakiteko.
     * @param line Egiaztatu beharreko testu lerroa.
     * @return True kopurua bikoitia bada (edo zero), false desorekatua bada.
     */
    private static boolean hasBalancedQuotes(String line) {
        long count = line.chars().filter(ch -> ch == '"').count();
        return count % 2 == 0;
    }
}
