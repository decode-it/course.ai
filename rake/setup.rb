namespace :setup do
    home_dir = ENV["HOME"]

    def apt_install(*names)
        names.flatten.each do |name|
            sh "sudo apt install #{name} -y"
        end
    end

    task :bootstrap do
        apt_install("git", "ruby-dev", "vim-gtk3")
    end

    task :sfml do
        apt_install(%w[libsfml-dev libudev-dev libopenal-dev libflac-dev libvorbis-dev libxrandr-dev libegl1-mesa-dev libxcb-image0-dev libjpeg-dev libfreetype6-dev freeglut3-dev])
    end

    task :gubg => [:bootstrap, :sfml] do
        ENV["gubg"] = "#{home_dir}/gubg" unless ENV["gubg"]
        Dir.chdir(home_dir) do
            if !File.read(".bashrc")["gubg"]
                File.open(".bashrc", "a") do |fo|
                    fo.puts("\n\n#GUBG environment setup")
                    fo.puts("export gubg=$HOME/gubg")
                    fo.puts("export PATH=$PATH:$gubg/bin")
                    fo.puts("export RUBYLIB=$gubg/ruby")
                end
            end
            if !File.exist?("gubg")
                sh "git clone https://github.com/gfannes/gubg"
                Dir.chdir("gubg") do
                    sh "git submodule update --init --recursive"
                    sh "rake uth"
                    sh "rake prepare"
                end
            end
        end
    end

    task :gcc do
        require("mkmf")
        if !find_executable("gcc-8")
            sh "sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y"
            sh "sudo apt update"
            apt_install("g++-8")
        end
        sh "sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10 --slave /usr/bin/g++ g++ /usr/bin/g++-8"
    end

    task :ninja do
        require("mkmf")
        if !find_executable("ninja")
            Dir.chdir(home_dir) do
                rm_rf "cook-binary"
                sh "git clone https://github.com/decode-it/cook-binary"
                if !File.read("#{home_dir}/.bashrc")["ninja"]
                    File.open("#{home_dir}/.bashrc", "a") do |fo|
                        fo.puts("\n\n#NINJA setup")
                        fo.puts("export PATH=$PATH:$HOME/cook-binary/ninja/linux")
                    end
                end
                if !ENV["PATH"]["ninja"]
                    ENV["PATH"] = "#{ENV["PATH"]}:#{ENV["HOME"]}/cook-binary/ninja/linux"
                end
            end
        end
    end

    task :cook => [:ninja, :gcc] do
        require("mkmf")
        if !find_executable("cook")
            Dir.chdir(home_dir) do
                rm_rf "cook"
                sh "git clone https://github.com/decode-it/cook"
                Dir.chdir("cook") do
                    sh "rake install"
                end
            end
        end
    end

    desc "Setup for Ubuntu 16"
    task :ub16 => [:gubg, :gcc, :cook]
end
