package com.idibon.ml.app

import scala.io.Source
import scala.util.Failure
import java.nio.file.FileSystems
import org.json4s._
import com.idibon.ml.train.Trainer

import com.idibon.ml.feature.bagofwords.{BagOfWordsTransformer, CaseTransform}
import com.idibon.ml.feature.indexer.IndexTransformer
import com.idibon.ml.feature.tokenizer.{TokenTransformer, Tag}
import com.idibon.ml.feature.language.LanguageDetector
import com.idibon.ml.feature.ngram.NgramTransformer
import com.idibon.ml.feature.{ContentExtractor, FeaturePipelineBuilder}
import com.typesafe.scalalogging.StrictLogging

import org.json4s.native.JsonMethods.parse

/** Simple command-line trainer tool
  *
  * Given a file containing aggregated training data, generates an
  * alloy and saves the alloy to an output file.
  */
object Train extends Tool with StrictLogging {

  private [this] def parseCommandLine(argv: Array[String]) = {
    val options = (new org.apache.commons.cli.Options)
      .addOption("i", "input", true, "Input file with training data")
      .addOption("o", "output", true, "Output alloy file")
      .addOption("r", "rules", true, "Input file with rules data")
      .addOption("w", "wiggle-wiggle", false, "Wiggle Wiggle")
      .addOption("n", "ngram", true, "Maximum n-gram size")

    new (org.apache.commons.cli.BasicParser).parse(options, argv)
  }

  def featurePipeline(ngramSize: Int) = {
    (FeaturePipelineBuilder.named("pipeline")
      += (FeaturePipelineBuilder.entry("convertToIndex", new IndexTransformer, "ngrams"))
      += (FeaturePipelineBuilder.entry("ngrams", new NgramTransformer(1, ngramSize), "bagOfWords"))
      += (FeaturePipelineBuilder.entry("bagOfWords",
        new BagOfWordsTransformer(List(Tag.Word, Tag.Punctuation), CaseTransform.ToLower),
        "convertToTokens", "languageDetector"))
      += (FeaturePipelineBuilder.entry("convertToTokens", new TokenTransformer, "contentExtractor", "languageDetector"))
      += (FeaturePipelineBuilder.entry("languageDetector", new LanguageDetector, "$document"))
      += (FeaturePipelineBuilder.entry("contentExtractor", new ContentExtractor, "$document"))
      := ("convertToIndex"))
  }

  def run(engine: com.idibon.ml.common.Engine, argv: Array[String]) {
    implicit val formats = org.json4s.DefaultFormats

    val cli = parseCommandLine(argv)
    val easterEgg = if (cli.hasOption('w')) Some(new WiggleWiggle()) else None
    easterEgg.map(egg => new Thread(egg).start)

    try{
      val startTime = System.currentTimeMillis()
      // default to tri-grams
      val ngramSize = Integer.valueOf(cli.getOptionValue('n', "3"))
      new Trainer(engine).train(featurePipeline(ngramSize),
        () => { // training data
          Source.fromFile(cli.getOptionValue('i'))
          .getLines.map(line => parse(line).extract[JObject])
        },
        () => { // rule data
          if (cli.hasOption('r')) {
            Source.fromFile(cli.getOptionValue('r'))
              .getLines.map(line => parse(line).extract[JObject])
          } else {
            List()
          }
        },
        None // option config
      ).map(alloy => alloy.save(cli.getOptionValue('o')))
        .map(x => {
          val elapsed = System.currentTimeMillis - startTime
          logger.info(s"Training completed in $elapsed ms")
        })
        .recoverWith({ case (error) => {
          logger.error("Unable to train model", error)
          Failure(error)
        }})
    } finally {
      easterEgg.map(_.terminate)
    }

  }

}
