package com.idibon.ml.cld2;

import java.util.List;
import java.util.HashMap;
import org.junit.*;
import static org.junit.Assert.*;
import static org.hamcrest.Matchers.*;
import static java.util.Arrays.asList;

public class CLD2Test {

    @BeforeClass public static void onlyOnce() {
        /* perform detection once to initialize the library; if
         * this fails, there's no point running the other tests */
        CLD2.detect("my voice is my passport, verify me",
                    CLD2.DocumentMode.PlainText);
    }

    private void run(HashMap<String, LangID> map) throws Exception {
        map.entrySet().forEach(entry -> {
            assertThat(CLD2.detect(entry.getKey(), CLD2.DocumentMode.PlainText),
                       is(equalTo(entry.getValue())));
        });
    }


    @Test public void testBasicLanguageID() throws Exception {
        // these strings were taken from the CLD2 test strings
        HashMap<String, LangID> tc = new HashMap<>();
        tc.put("баба цои уара бара уаба нхои баба", LangID.ABKHAZIAN);
        tc.put("het om vir die Cowboys in Amerika", LangID.AFRIKAANS);
        tc.put("bly dat ons kon stap en dat ons uit", LangID.AFRIKAANS);
        tc.put("الرزق والغنى وقضاء ", LangID.ARABIC);
        tc.put("Ferit greu un casteller de la colla", LangID.CATALAN);
        tc.put("Semne ca vrea sa treaca la sex", LangID.ROMANIAN);
        tc.put("Daghang mga tawo ang gindugo gikan sa almoranas", LangID.CEBUANO);
        tc.put("ᎤᏂᏩᎯᏍᏗ ᎦᏙ ᏓᏂᏍᏕᎵᏍᎬ ᎾᎥ ᏂᏚᎾᏓᎴᏫᏒ", LangID.CHEROKEE);
        tc.put("Vi scatineti contr'a Cupruduzzione", LangID.CORSICAN);
        tc.put("lovforslag, som igen gør det", LangID.DANISH);
        tc.put("auge , auf das neben dem", LangID.GERMAN);
        tc.put("θέματος - Τροχοί", LangID.GREEK);
        tc.put("that 123people does not verify the", LangID.ENGLISH);
        tc.put("tous nos Qu'est ce qu'un", LangID.FRENCH);
        tc.put("de fraach nei te pakken", LangID.FRISIAN);
        tc.put("Shocraigh muid a dhéanamh", LangID.IRISH);
        tc.put("פשוט נהדר. העור במשך שבוע ~ ", LangID.HEBREW);
        tc.put("현재위치로 쉽게 이동 가능합니", LangID.KOREAN);
        tc.put("出ようと思ってるんだ", LangID.JAPANESE);
        tc.put("טיאָן פלייץ גרופּעס טראַנ", LangID.YIDDISH);
        tc.put("后打时他们的同事", LangID.CHINESE);
        run(tc);
    }

    @Test public void testMixedLanguageID() throws Exception {
        // these strings were taken from the CLD2 test strings
        HashMap<String, LangID> tc = new HashMap<>();
        tc.put("баба цои уара " +
               "het om vir die Cowboys in Amerika", LangID.AFRIKAANS);
        tc.put("that 123people does not verify " +
               "出ようと", LangID.ENGLISH);
        tc.put("出ようと " +
               "ᎤᏂᏩᎯᏍᏗ " +
               "思ってるんだ", LangID.JAPANESE);
        tc.put("Semne ca " +
               "auf das neben dem", LangID.GERMAN);
        tc.put("that 123people does " +
               "auf das " +
               "后打时他们" +
               "not verify", LangID.ENGLISH);
        run(tc);
    }

    @Test public void testWithEmoji() throws Exception {
        HashMap<String, LangID> tc = new HashMap<>();
        tc.put("Some english text with emoji: \ud83d\udca9", LangID.ENGLISH);
        tc.put("\u26c4 het om vir die Cowboys in Amerika \ud83c\uddfa\ud83c\uddf8", LangID.AFRIKAANS);
        tc.put("日本語\ud83c\uddef\ud83c\uddf5", LangID.CHINESE_T);
        tc.put("日本語です\ud83c\uddef\ud83c\uddf5", LangID.JAPANESE);
        run(tc);
    }

    @Test public void testWithEmoticons() throws Exception {
        HashMap<String, LangID> tc = new HashMap<>();
        tc.put("Whatevs... ¯\\_(ツ)_/¯", LangID.ENGLISH);
        tc.put("Raise your dongers ヽ༼ຈل͜ຈ༽ﾉ", LangID.ENGLISH);
        tc.put("༼ つ ◕_◕ ༽つ please give!", LangID.ENGLISH);
        run(tc);
    }

    @Test public void testMisidentified() throws Exception {
        HashMap<String, LangID> tc = new HashMap<>();
        tc.put("¯\\_(ツ)_/¯", LangID.JAPANESE);
        tc.put("༼ つ ◕_◕ ༽つ", LangID.JAPANESE);
        tc.put("fu >:(", LangID.TONGA);
        tc.put("ヽ༼ຈل͜ຈ༽ﾉ", LangID.ARABIC);
        run(tc);
    }

    @Test public void testUnknownCases() throws Exception {
        HashMap<String, LangID> tc = new HashMap<>();
        tc.put("\ud83d\udca9", LangID.UNKNOWN_LANGUAGE);
        tc.put("\ud83d\udc6e \ud83d\udd28", LangID.UNKNOWN_LANGUAGE);
        tc.put("", LangID.UNKNOWN_LANGUAGE);
        tc.put("3.1415926538", LangID.UNKNOWN_LANGUAGE);
        tc.put("#!$?", LangID.UNKNOWN_LANGUAGE);
        tc.put(":)", LangID.UNKNOWN_LANGUAGE);
        run(tc);
    }
}
