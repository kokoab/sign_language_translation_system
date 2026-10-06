require 'xcodeproj'
root = File.join(__dir__, 'phone')
project = Xcodeproj::Project.new(File.join(root, 'ClassifierBench.xcodeproj'))
app = project.new_target(:application, 'ClassifierBench', :ios, '15.0')
test = project.new_target(:unit_test_bundle, 'ClassifierBenchTests', :ios, '15.0')
test.add_dependency(app)
[app, test].each do |target|
 target.build_configurations.each do |c|
  c.build_settings['DEVELOPMENT_TEAM'] = '2BJ6GHCUJ2'
  c.build_settings['CODE_SIGN_STYLE'] = 'Automatic'
  c.build_settings['GENERATE_INFOPLIST_FILE'] = 'YES'
  c.build_settings['SWIFT_VERSION'] = '5.0'
  c.build_settings['TARGETED_DEVICE_FAMILY'] = '1'
  c.build_settings['PRODUCT_BUNDLE_IDENTIFIER'] = target == app ? 'com.kokoab.atlasClassifierBench' : 'com.kokoab.atlasClassifierBench.tests'
  if target == test
   c.build_settings['TEST_HOST'] = '$(BUILT_PRODUCTS_DIR)/ClassifierBench.app/ClassifierBench'
   c.build_settings['BUNDLE_LOADER'] = '$(TEST_HOST)'
  else
   c.build_settings['INFOPLIST_KEY_UILaunchScreen_Generation'] = 'YES'
   c.build_settings['INFOPLIST_KEY_CFBundleDisplayName'] = 'ATLAS Bench'
  end
 end
end
app.source_build_phase.add_file_reference(project.main_group.new_file('App.swift'))
test.source_build_phase.add_file_reference(project.main_group.new_file('BenchTests.swift'))
Dir.glob(File.join(root,'Resources','*')).sort.each do |path|
 ref = project.main_group.new_file('Resources/' + File.basename(path))
 if path.end_with?('.mlpackage')
  app.source_build_phase.add_file_reference(ref)
 else
  app.resources_build_phase.add_file_reference(ref)
 end
end
project.save
scheme = Xcodeproj::XCScheme.new
scheme.configure_with_targets(app,test)
scheme.save_as(project.path,'ClassifierBench',true)
puts project.path
