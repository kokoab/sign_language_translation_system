require "xcodeproj"
p=Xcodeproj::Project.open(ARGV[0])
t=p.targets.find{|x| x.name=="Runner"}
ARGV[1..].each do |path|
 r=p.main_group.new_reference(path)
 r.source_tree="<absolute>"
 r.path=path
 r.last_known_file_type="folder.mlpackage"
 t.source_build_phase.add_file_reference(r)
end
p.save
