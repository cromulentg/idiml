package com.idibon.ml.feature.tokenizer

import com.ibm.icu
import org.scalatest.{Matchers, FunSpec}

class HTMLBreakIteratorSpec extends FunSpec
    with Matchers with BreakIteratorHelpers[HTMLBreakIterator] {

  def newIterator(del: icu.text.BreakIterator) = new HTMLBreakIterator(del)

  it("should support empty strings") {
    !!("") shouldBe List("")
  }

  it("should put entire script sections into single tokens") {
    !!("""<script type="text/javascript">var do = function(i) {}</script>foo""") shouldBe List(
      """<script type="text/javascript">var do = function(i) {}</script>""", "foo")
  }

  it("should put entire style sections into single tokens") {
    !!("""<head><style type="text/css">body {
color:red;
}
</style></head><body>text</body>""") shouldBe List("<head>",
      """<style type="text/css">body {
color:red;
}
</style>""", "</head>", "<body>", "text", "</body>")
  }

  it("should be case insensitive") {
    !!("<Script></scripT >") shouldBe List("<Script></scripT >")
  }

  it("should treat self-closing tags as boundaries") {
    !!("""<script src="/foo.html"/>function(i) {}</script>""") shouldBe List(
      """<script src="/foo.html"/>""", "function", "(", "i", ")",
      " ", "{", "}", "</script>")
  }
}
