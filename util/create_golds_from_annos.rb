# This script takes existing annotations from IdiML training data and converts
# them into golds for evaluation. Because training data includes only content,
# language code, and annotations, no other metadata is included beyond the gold
# label. This script assumes mutually-exclusive labels with a single positive
# annotation per document. New train and test sets will be generated in the 
# same directory as the source file.
#
# author: "Nicholas Gaylord | nick@idibon.com"
#
# source_annos is the name of the existing training data
# label_mapping is the file output by training containing rules and uuid
# 	to human-readable name mappings
# holdout_percent is a value between 0 and 1, representing an approximate
# 	percentage of the source data to be held out as testing golds


require "json"

source_annos = ARGV[0]
label_map_file = ARGV[1]
holdout_percent = ARGV[2]

gold_file = source_annos.to_s+"_"+holdout_percent.to_s+"_golds.json"
train_file = source_annos.to_s+"_"+holdout_percent.to_s+"_train.json"

File.open(gold_file, "w") {}
File.open(train_file, "w") {}

label_map = {}
File.open(label_map_file) do |f|
	label_map = JSON.parse(f.read)
end

doc_name = 0

File.foreach(source_annos) do |doc|
	rand_val = rand
	if rand_val <= holdout_percent.to_f
		parsed = JSON.parse(doc)
		content = parsed["content"]
		annos = parsed["annotations"]
		gold_label = ""
		gold_uuid = ""
		annos.each { |anno|
			if anno["isPositive"] == true
				gold_uuid = anno["label"]["name"]
				gold_label = label_map["uuid_to_label"][gold_uuid]
			end
		}
		gold_info = {"content" => content, "name" => doc_name.to_s,
								 "metadata" => { "gold" => gold_label,
								 "gold_uuid" => gold_uuid } }
	 	doc_name += 1	
		File.open(gold_file, "a") { |f| f.write(gold_info.to_json+"\n")
		}
	else
		File.open(train_file, "a") { |f| f.write(doc)
		}
  end
end
