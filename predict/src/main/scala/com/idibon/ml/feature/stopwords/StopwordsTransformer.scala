package com.idibon.ml.feature.stopwords

import java.net.URI
import java.util.Properties

import com.ibm.icu.text.Normalizer2
import com.idibon.ml.alloy.Alloy.{Writer, Reader}
import com.idibon.ml.alloy.Codec
import com.idibon.ml.common.{Archivable, Engine, ArchiveLoader}
import com.idibon.ml.feature.{FeatureTransformer, Feature}
import com.idibon.ml.feature.bagofwords.{CaseTransform, Word}
import com.idibon.ml.feature.language.LanguageCode
import com.idibon.ml.feature.tokenizer.{Tag, Token}
import org.json4s._
import scala.collection.JavaConverters._
import scala.collection.immutable.IndexedSeq
import scala.io.Source

/**
  * Created by nick gaylord on 3/18/16.
  *
  * Stopwords are frequent tokens, listed in separate language-specific files,
  * that model builders may want to exclude from being counted as unigram
  * features. This transformer takes a sequence of Words and returns another
  * sequence of Words filtered to exclude the contents of the stopword file
  * corresponding to the document's LanguageCode
  *
  * @param stopwordMap map of three letter iso code to set of words
  * @param defaultStopwordLanguage the default language to use if none is detected in the content
  *                                that is passed in.
  */
class StopwordsTransformer(private[stopwords] val stopwordMap: Map[String, Set[String]],
                           val defaultStopwordLanguage: String = "")
  extends FeatureTransformer
  with Archivable[StopwordsTransformer, StopwordsTransformerLoader] {

  /**
    * Word-by-word, identify as a stopword or not
    *
    * @param x -- an individual word
    * @param stops -- the stopword list corresponding to the document's language code
    * @return -- Boolean, 1 if the word is not in the stopword list, 0 if it is
    */
  def filter(x: Word, stops: Set[String]) = {
    !stops.contains(x.word)
  }

  /**
    * Take a sequence of words (a document) and return a sequence of words
    * with stopwords removed.
    *
    * @param input -- a sequence of Words
    * @param lc -- the document's LanguageCode
    * @return -- a sequence of Words that is a (not necessarily proper) subset of input
    */
  def apply(input: Seq[Word], lc: Feature[LanguageCode]): Seq[Word] = {
    val code: LanguageCode = lc.get
    val stopSet = stopwordMap.getOrElse(code.iso_639_2.getOrElse(defaultStopwordLanguage), Set())
    input.filter(x => filter(x, stopSet))
  }

  /** Serializes the object within the Alloy
    *
    * Implementations are responsible for persisting any internal state
    * necessary to re-load the object (for example, feature-to-vector
    * index mappings) to the provided Alloy.Writer.
    *
    * Implementations may return a JObject of configuration data
    * to include when re-loading the object.
    *
    * @param writer destination within Alloy for any resources that
    *               must be preserved for this object to be reloadable
    * @return Some[JObject] of configuration data that must be preserved
    *         to reload the object. None if no configuration is needed
    */
  override def save(writer: Writer): Option[JObject] = {
    // find stop words to save that are different from our defaults
    val baseStopwords = StopwordsTransformer.generateStopFileMap()
    val stopWordsToSave = stopwordMap.flatMap({case (lang, words) =>
      val defaultWords = baseStopwords.getOrElse(lang, Set[String]())
      if (words.equals(defaultWords)) {
        None
      } else {
        Some((lang, words))
      }
    })
    // write them out -- first what languages
    val baseDir = writer.within(StopwordsTransformer.STOPWORD_RESOURCE_NAME)
    val languages = baseDir.resource(StopwordsTransformer.STOPWORD_LANGUAGE_RESOURCE_NAME)
    Codec.VLuint.write(languages, stopWordsToSave.size)
    stopWordsToSave.foreach({case (code, _) =>
      Codec.String.write(languages, code)
    })
    languages.close()
    // second the actual words into each language resource
    stopWordsToSave.foreach({case (code, words) =>
      val resource = baseDir.resource(code)
      Codec.VLuint.write(resource, words.size)
      words.foreach(w => Codec.String.write(resource, w))
      resource.close()
    })
    val resource = baseDir.resource(StopwordsTransformer.STOPWORD_LANGUAGE_DEFAULT)
    Codec.String.write(resource, defaultStopwordLanguage)
    resource.close()
    None
  }
}

sealed case class LowerCaseTransform() extends CaseTransform {
  val transform = CaseTransform.ToLower
}

sealed case class UpperCaseTransform() extends CaseTransform {
  val transform = CaseTransform.ToUpper
}

/**
  * Companion object that houses static methods to get started.
  */
object StopwordsTransformer {
  val languageIndex = "stopwords.properties"
  val STOPWORD_LANGUAGE_RESOURCE_NAME = "languages"
  val STOPWORD_RESOURCE_NAME = "stopwords"
  val STOPWORD_LANGUAGE_DEFAULT = "language_default.txt"
  val lower = new LowerCaseTransform()
  val upper = new UpperCaseTransform()
  /* See http://unicode.org/reports/tr15/#Norm_Forms for the different types.
     We want the one that takes precombined chars to decomposed chars */
  val icu4jnormalizer = Normalizer2.getNFDInstance()

  /**
    * Iterate over a list of stopword files and extract their contents,
    * normalizing filenames to three-letter ISO 639-2 language codes
    *
    * @return -- a map from language codes to language-specific stopword lists
    */
  def generateStopFileMap() = {
    val props = new Properties()
    props.load(getClass.getClassLoader.getResourceAsStream(StopwordsTransformer.languageIndex))
    val codes = props.stringPropertyNames().asScala

    codes.map(threeLetterCode => {
      val resource = props.getProperty(threeLetterCode)
      val stopwordFile = getClass.getClassLoader.getResourceAsStream(resource)
      val stopwordStream = Source.fromInputStream(stopwordFile)
      val stopwordLines = stopwordStream.getLines().toSeq
      val stopwordSet = createStopwordsSet(stopwordLines, Some(threeLetterCode))
      (threeLetterCode, stopwordSet)
    }).toMap
  }

  /**
    * Creates a stopword set from the passed in sequence of strings and language code.
    *
    * It creates upper & lower x normalized & non-normalized versions. If it is already
    * normalized on the way in, then we drop the non-normalized versions.
    *
    * @param lines the sequence to use for stop words.
    * @param lc the language code if known.
    * @return a set of stopwords.
    */
  def createStopwordsSet(lines:Seq[String], lc:Option[String]) = {
    lines.flatMap(l => {
      val t = new Token(l, Tag.Word, 0, 0)
      val lowerF = lower.transformation(new LanguageCode(lc))
      val upperF = upper.transformation(new LanguageCode(lc))
      val tLower = lowerF(t)
      val tUpper = upperF(t)
      Seq(tUpper, tLower)
    }).flatMap(s => Seq(s.word, icu4jnormalizer.normalize(s.word)))
      .toSet
  }
}

class StopwordsTransformerLoader extends ArchiveLoader[StopwordsTransformer] {

  /** Reloads the object from the Alloy
    *
    * @param engine implementation of the Engine trait
    * @param reader location within Alloy for loading any resources
    *               previous preserved by a call to
    *               { @link com.idibon.ml.common.Archivable#save}
    * @param config archived configuration data returned by a previous
    *               call to { @link com.idibon.ml.common.Archivable#save}
    * @return this object
    */
  override def load(engine: Engine,
                    reader: Option[Reader],
                    config: Option[JObject]): StopwordsTransformer = {
    val defaultStopwords = StopwordsTransformer.generateStopFileMap()
    var defaultLanguage = ""
    val stopwordMap: Map[String, Set[String]] = reader match {
      case None => {
        // check config to see if we are to load any custom stop words
        config match {
          case None => defaultStopwords
          case Some(config) => {
            implicit val formats = DefaultFormats
            val languageConfig = config.extract[StopwordsConfig]
            defaultLanguage = languageConfig.defaultStopwordLanguage
            val customStopwords = loadExternalStopwords(languageConfig)
            createStopwordMap(customStopwords, defaultStopwords)
          }
        }
      }
      case Some(reader) => {
        //read vocab from alloy
        val baseDir = reader.within(StopwordsTransformer.STOPWORD_RESOURCE_NAME)
        val resource = baseDir.resource(StopwordsTransformer.STOPWORD_LANGUAGE_DEFAULT)
        if (resource != null) {
          defaultLanguage = Codec.String.read(resource)
        }
        val languageList = readLanguages(baseDir)
        val savedStopwords = readStopwordMap(baseDir, languageList)
        createStopwordMap(savedStopwords, defaultStopwords)
      }
    }
    new StopwordsTransformer(stopwordMap, defaultLanguage)
  }

  /**
    * Helper method to load external stopwords.
    *
    * Currently we only support loading a local file.
    *
    * @param languageConfig contains mapping of language -> URI
    * @return a map of stopwords
    */
  def loadExternalStopwords(languageConfig: StopwordsConfig): Map[String, Set[String]] = {
    languageConfig.languages.map({ case (lang, uriString) =>
      val uri = new URI(uriString)
      val stopwords = uri.getScheme match {
        case "file" => Source.fromURI(uri).getLines().toSet
        case _ => throw new IllegalArgumentException(
          s"Stopword file scheme ${uri.getScheme} is not currently supported.")
      }
      (lang, stopwords)
    })
  }

  /**
    * Helper method to create a stop word map from a custom & default stop word list.
    *
    * @param customStopwords the custom stopwords to use inplace over any default stop words.
    * @param defaultStopwords the default stopwords to fall back to.
    * @return stopword map to use.
    */
  def createStopwordMap(customStopwords: Map[String, Set[String]],
                        defaultStopwords: Map[String, Set[String]]): Map[String, Set[String]] = {
    defaultStopwords.keySet.union(customStopwords.keySet).map(lang => {
      (lang, customStopwords.getOrElse(lang, defaultStopwords.getOrElse(lang, Set[String]())))
    }).toMap
  }

  /**
    * Reads the stopword map from the alloy.
    *
    * @param baseDir
    * @param languageList
    * @return
    */
  def readStopwordMap(baseDir: Reader, languageList: IndexedSeq[String]): Map[String, Set[String]] = {
    languageList.map(code => {
      val resource = baseDir.resource(code)
      val numWords = Codec.VLuint.read(resource)
      val words = (0 until numWords).map(_ => {
        Codec.String.read(resource)
      })
      resource.close()
      (code, words.toSet)
    }).toMap
  }

  /**
    * Helper method to read the languages that are in this stop word set.
    *
    * @param baseDir
    * @return
    */
  def readLanguages(baseDir: Reader) = {
    val languageResource = baseDir.resource(StopwordsTransformer.STOPWORD_LANGUAGE_RESOURCE_NAME)
    val numLanguages = Codec.VLuint.read(languageResource)
    val languageList = (0 until numLanguages).map(_ => {
      Codec.String.read(languageResource)
    })
    languageResource.close()
    languageList
  }
}

/** JSON config class. Only used during training. **/
sealed case class StopwordsConfig(languages: Map[String, String], defaultStopwordLanguage: String = "")
