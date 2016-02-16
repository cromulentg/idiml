package com.idibon.ml.feature.word2vec

import java.util
import java.net.URI

import org.scalatest._

import scala.collection.JavaConversions.mapAsScalaMap

class Word2VecBinReaderSpec extends FunSpec with Matchers {

  describe("A bin file read in with Word2VecBinReader") {

    val reader = new Word2VecBinReader
    val binFileUriString= "file://" + System.getProperty("user.dir") + "/src/test/resources/fixtures/word2vec-test-vectors.bin.gz"
    val word2VecMap: util.LinkedHashMap[String, Array[Float]] = reader.parseBinFile(new URI(binFileUriString))


    it("should have the expected vocabulary") {
      val expectedVocab = Set("</s>", "the", "of", "and", "in", "to", "a", "as", "is", "one", "that", "with", "or", "autism", "anarchism", "s", "for", "autistic", "are", "anarchists", "zero", "anarchist", "by", "nine", "be", "not", "it", "some", "people", "they", "an", "have", "many", "this", "from", "was", "two", "social", "on", "eight", "such", "other", "their", "at", "also", "can", "often", "which", "has", "most", "see", "anarcho", "may", "three", "more", "movement", "who", "there", "state", "first", "disorder", "children", "what", "his", "language", "these", "but", "society", "its", "he", "six", "groups", "child", "syndrome", "movements", "use", "anti", "than", "them", "early", "all", "seven", "property", "asperger", "been", "however", "ideas", "form", "do", "post", "term", "used", "individuals", "others", "war", "those", "against", "although", "authoritarian", "anarchy", "so", "proudhon", "self", "were", "four", "would", "within", "general", "five", "capitalism", "person", "autistics", "about", "will", "workers", "when", "cnt", "violence", "high", "like", "different", "even", "non", "disorders", "while", "no", "called", "bakunin", "example", "development", "communication", "functioning", "revolution", "work", "sometimes", "number", "lack", "united", "both", "feminist", "age", "developmental", "class", "means", "believe", "individual", "since", "being", "over", "years", "much", "system", "due", "if", "include", "spanish", "world", "left", "culture", "certain", "least", "behavior", "including", "without", "political", "philosophy", "syndicalism", "group", "another", "feminism", "mental", "diagnosis", "spectrum", "normal", "should", "based", "kropotkin", "had", "century", "communism", "civil", "right", "i", "thought", "international", "nature", "must", "industrial", "action", "popular", "common", "anarcha", "she", "out", "because", "interaction", "part", "cure", "adults", "able", "sensory", "rett", "working", "way", "up", "economic", "might", "argue", "considered", "modern", "during", "time", "major", "t", "labor", "own", "individualist", "community", "her", "revolutionary", "after", "into", "communists", "christian", "spain", "government", "interests", "diagnostic", "symptoms", "skills", "dsm", "pervasive", "problems", "body", "french", "word", "institutions", "rather", "association", "free", "how", "before", "human", "every", "true", "described", "american", "libertarian", "published", "later", "point", "known", "theory", "only", "tucker", "warren", "new", "between", "activity", "communist", "need", "us", "goldman", "further", "leo", "associated", "members", "national", "life", "fascist", "just", "authority", "schools", "specific", "following", "list", "repetitive", "level", "speech", "student", "help", "english", "still", "any", "violent", "related", "advocate", "does", "structures", "history", "principles", "labour", "similar", "th", "europe", "religious", "man", "de", "where", "famous", "opposed", "supported", "interest", "set", "natural", "through", "take", "conflict", "communities", "liberty", "links", "important", "late", "force", "tolstoy", "g", "prior", "today", "led", "states", "fascism", "rise", "months", "well", "environment", "school", "particular", "present", "single", "information", "article", "view", "intellectual", "ability", "criteria", "conditions", "delays", "play", "communicate", "kanner", "typical", "learning", "childhood", "trouble", "speak", "little", "students", "increase", "label", "belief", "interpretations", "particularly", "mutual", "law", "william", "godwin", "rothbard", "found", "sovereignty", "book", "until", "complete", "rights", "involved", "trade", "france", "stirner", "power", "long", "benjamin", "became", "theorists", "marx", "emma", "syndicalist", "below", "several", "same", "black", "institute", "n", "cgt", "significant", "bolsheviks", "control", "themselves", "develop", "fascists", "areas", "kingdom", "physical", "focus", "views", "green", "day", "books", "capitalist", "terms", "controversial", "throughout", "idea", "reference", "small", "referred", "cause", "cultural", "living", "recent", "health", "voting", "patterns", "environmental", "diagnosed", "statistical", "manual", "dr", "fact", "iv", "stereotyped", "behaviors", "things", "seem", "difficulty", "appear", "parents", "research", "unusual", "understand", "difficulties", "given", "become", "better", "public", "iq")
      word2VecMap.keys shouldBe expectedVocab
    }

    it("should have the expected vector for the word 'the'") {
      val the_vec = word2VecMap.get("the")
      val expectedSampleVector = Array(0.002279663f,-0.0049525453f,0.0043148804f,4.7988893E-4f,-0.0018310547f,-0.0020832825f,-7.4798585E-4f,-0.0029350282f,-0.0043695066f,-0.0028370665f,0.0042770384f,-0.0017698669f,-0.0026794435f,0.0023173522f,-0.0017037964f,0.0047566225f,-0.0016046143f,-0.004588776f,0.0027157592f,-7.43103E-5f,-0.0034887695f,9.757996E-4f,0.0014419556f,-4.8141478E-4f,0.0018243408f,-0.003957672f,8.810425E-4f,0.0018165589f,-0.0030090332f,1.4205933E-4f,-6.072998E-5f,-0.0023991393f,-0.0028326416f,-0.004693756f,2.2888184E-5f,1.5274048E-4f,9.765625E-6f,-0.0039338684f,-0.004961853f,-0.0047465516f,-0.0043255617f,-5.4702756E-4f,0.0013912964f,0.0011842346f,-0.003182373f,0.004998016f,-0.0020114137f,-0.0012736511f,-0.001404419f,0.0047325133f,-0.003763733f,0.0011610412f,-0.0013354492f,0.0031877137f,4.058838E-5f,0.0042695617f,-0.0028192138f,-0.002605133f,-0.0041921996f,-0.0036668396f,-0.0031994628f,-0.0031147767f,0.0024441527f,-0.0018669128f,0.0026800537f,0.003690033f,0.001355896f,0.002950592f,0.002475586f,0.0023405456f,-0.0035507202f,-0.003433075f,-0.0036566162f,-1.3198852E-4f,0.0041305544f,-0.002736664f,-0.0030603027f,-0.0041963197f,0.0033059693f,-0.0041789245f,-5.792236E-4f,0.0021788024f,-0.004618225f,-0.0044786073f,0.001442871f,0.0035246278f,0.004264221f,0.0021455383f,0.0031622315f,-0.003127594f,-0.003640442f,0.003974762f,-0.0027648925f,0.0017533875f,5.740356E-4f,0.0017903137f,-0.001182251f,1.9882202E-4f,-0.0016860962f,-0.0011265564f)
      the_vec shouldBe expectedSampleVector
    }

    it("should only have keys which are arrays of floats of length 100"){
      for ((k,v) <- word2VecMap) { v should have length 100 }
    }

  }
}
