#Make sure you have rake installed
#Ubuntu: sudo apt install rake

require("./rake/setup.rb")

def cooker()
    require("gubg/shared")
    require("gubg/build/Cooker")
    c = GUBG::Build::Cooker.new
    c.output(".build")
    case GUBG::os
    when :windows then c.option("c++.std", 14)
    else c.option("c++.std", 17) end
    c
end

desc "Build the AI app"
task :build do
    cooker().option("release").generate(:ninja, "ai/gen").ninja().run()
    cooker().option("release").generate(:ninja, "ai/app").ninja().run()
end

desc "Create documentation"
task :doc do
    sh "pandoc -o exercises.pdf doc/exercises.md"
    sh "evince exercises.pdf"
end

desc "clean"
task :clean do
    rm FileList.new("*.naft")
    rm FileList.new("ai.*")
    rm_rf ".cook"
    rm_rf ".build"
end

task :default do
    sh "rake -T"
end
