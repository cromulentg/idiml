# This script takes existing annotations from IdiML training data and converts
# them into golds for evaluation. Because training data includes only content,
# language code, and annotations, no other metadata is included beyond the gold
# label. This script assumes mutually-exclusive labels with a single positive
# annotation per document.
#
# source_annos is the name of the existing training data
# holdout_percent is a value between 0 and 1, representing an approximate
# 	percentage of the source data to be held out as testing golds

require "json"

source_annos = ARGV[0]
holdout_percent = ARGV[1]

File.open(source_annos.to_s+"_"+holdout_percent.to_s+"_golds.json", "w") {}
File.open(source_annos.to_s+"_"+holdout_percent.to_s+"_train.json", "w") {}

File.foreach(source_annos) do |doc|
	rand_val = rand
	if rand_val <= holdout_percent.to_f
		parsed = JSON.parse(doc)
		content = parsed["content"]
		annos = parsed["annotations"]
		gold_label = ""
		annos.each { |anno|
			if anno["isPositive"] == true
				gold_label = anno["label"]["name"]
			end
		}
		gold_info = {"content" => content, "metadata" => { "gold" => gold_label}}
		File.open(source_annos.to_s+"_"+holdout_percent.to_s+"_golds.json", "a") { |f|
			f.write(gold_info.to_json+"\n")
		}
	else
		File.open(source_annos.to_s+"_"+holdout_percent.to_s+"_train.json", "a") { |f|
			f.write(doc)
		}
  end
end
