"""
Curated top-repository list per Sugar High language.

Chosen for: high star count, idiomatic production code, permissive licenses
(MIT / Apache-2.0 / BSD / ISC / MPL / Unlicense-ish), and enough volume of the
target language to hit the per-language token budget on its own.

Each entry is "owner/name". Tarballs are fetched from codeload at HEAD.
"""

REPOS: dict[str, list[str]] = {
    'javascript': [
        'expressjs/express', 'lodash/lodash', 'axios/axios', 'reduxjs/redux',
        'chartjs/Chart.js', 'd3/d3', 'mrdoob/three.js', 'vuejs/core',
        'facebook/react', 'babel/babel', 'webpack/webpack', 'eslint/eslint',
        'facebook/jest', 'hexojs/hexo', 'koajs/koa', 'socketio/socket.io',
        'gulpjs/gulp', 'markedjs/marked',
    ],
    'typescript': [
        'microsoft/vscode', 'nestjs/nest', 'trpc/trpc', 'pmndrs/zustand',
        'colinhacks/zod', 'TanStack/query', 'vitejs/vite', 'excalidraw/excalidraw',
        'calcom/cal.com', 'payloadcms/payload', 'ant-design/ant-design',
        'storybookjs/storybook', 'prisma/prisma', 'clerk/javascript',
        'refinedev/refine',
    ],
    'python': [
        'pallets/flask', 'psf/requests', 'django/django', 'pandas-dev/pandas',
        'encode/httpx', 'pydantic/pydantic', 'fastapi/fastapi', 'scrapy/scrapy',
        'scikit-learn/scikit-learn', 'home-assistant/core', 'celery/celery',
        'pallets/werkzeug', 'pallets/jinja', 'pallets/click', 'python/mypy',
        'certbot/certbot', 'networkx/networkx', 'sympy/sympy',
    ],
    'rust': [
        'BurntSushi/ripgrep', 'sharkdp/bat', 'tokio-rs/tokio', 'serde-rs/serde',
        'clap-rs/clap', 'rust-lang/regex', 'starship/starship', 'sharkdp/fd',
        'actix/actix-web', 'hyperium/hyper', 'tauri-apps/tauri', 'diesel-rs/diesel',
    ],
    'go': [
        'gin-gonic/gin', 'spf13/cobra', 'sirupsen/logrus', 'gorilla/mux',
        'go-chi/chi', 'urfave/cli', 'stretchr/testify', 'hashicorp/consul',
        'moby/moby', 'kubernetes/kubernetes', 'caddyserver/caddy', 'fatih/color',
    ],
    'c': [
        'curl/curl', 'redis/redis', 'git/git', 'jqlang/jq',
        'libuv/libuv', 'nothings/stb', 'antirez/sds', 'DaveGamble/cJSON',
        'openssl/openssl', 'jemalloc/jemalloc',
        'sqlite/sqlite', 'facebook/zstd', 'lz4/lz4', 'wg/wrk',
        'file/file',
    ],
    'cpp': [
        'nlohmann/json', 'fmtlib/fmt', 'google/googletest', 'gabime/spdlog',
        'catchorg/Catch2', 'protocolbuffers/protobuf', 'opencv/opencv', 'ocornut/imgui',
        'aseprite/aseprite', 'godotengine/godot', 'aria2/aria2',
    ],
    'csharp': [
        'dotnet/runtime', 'AutoMapper/AutoMapper', 'JamesNK/Newtonsoft.Json',
        'serilog/serilog', 'App-vNext/Polly', 'restsharp/RestSharp',
        'quartznet/quartznet', 'MassTransit/MassTransit',
        'dotnet/roslyn', 'jellyfin/jellyfin', 'RavenDB/ravendb',
        'AvaloniaUI/Avalonia', 'CommunityToolkit/dotnet',
    ],
    'java': [
        'google/guava', 'square/retrofit', 'square/okhttp', 'ReactiveX/RxJava',
        'alibaba/fastjson', 'spring-projects/spring-framework', 'apache/commons-lang',
        'junit-team/junit5', 'spring-projects/spring-boot', 'elastic/elasticsearch',
        'apache/dubbo', 'mybatis/mybatis-3', 'google/error-prone', 'apache/flink',
        'apache/lucene', 'quarkusio/quarkus',
        'apache/kafka', 'apache/hadoop', 'apache/cassandra',
        'apache/tomcat', 'eclipse-vertx/vert.x',
    ],
    'kotlin': [
        'square/okio', 'InsertKoinIO/koin', 'Kotlin/kotlinx.coroutines',
        'Kotlin/kotlinx.serialization', 'arrow-kt/arrow', 'ktorio/ktor',
        'JetBrains/Exposed', 'mockk/mockk',
    ],
    'swift': [
        'Alamofire/Alamofire', 'SwiftyJSON/SwiftyJSON', 'apple/swift-nio',
        'ReactiveX/RxSwift', 'vapor/vapor', 'pointfreeco/swift-composable-architecture',
        'ivanvorobei/SwiftUI', 'exyte/PopupView', 'Dimillian/IceCubesApp',
        'sindresorhus/Actions', 'MonitorControl/MonitorControl', 'lwouis/alt-tab-macos',
        'p0deje/Maccy', 'gao-sun/eul', 'stephencelis/SQLite.swift', 'kean/Nuke',
    ],
    'ruby': [
        'rails/rails', 'sinatra/sinatra', 'jekyll/jekyll', 'rubocop/rubocop',
        'fastlane/fastlane', 'rspec/rspec-core', 'sidekiq/sidekiq', 'ruby/rake',
        'heartcombo/devise', 'caskroom/homebrew-cask',
        'puma/puma', 'Homebrew/brew', 'gitlabhq/gitlabhq', 'chef/chef',
    ],
    'php': [
        'laravel/framework', 'symfony/symfony', 'guzzle/guzzle', 'composer/composer',
        'phpunit/phpunit', 'monolog/monolog', 'slimphp/Slim', 'doctrine/orm',
        'filamentphp/filament', 'briannesbitt/Carbon', 'spatie/laravel-permission',
        'fakerphp/faker', 'tymondesigns/jwt-auth',
        'WordPress/WordPress', 'phpmyadmin/phpmyadmin', 'top-think/framework',
    ],
    'lua': [
        'openresty/lua-nginx-module', 'nvim-lua/kickstart.nvim', 'kikito/inspect.lua',
        'rxi/json.lua', 'nvim-telescope/telescope.nvim', 'hoelzro/lua-term',
        'lunarmodules/luassert', 'LuaLS/lua-language-server', 'lunarmodules/busted',
        'nvim-lualine/lualine.nvim', 'lewis6991/gitsigns.nvim', 'kikito/middleclass',
        'love2d-community/awesome-love2d', 'Kong/kong', 'leafo/moonscript',
        'nvim-tree/nvim-tree.lua',
    ],
    'zig': [
        'ziglang/zig', 'oven-sh/bun', 'zigtools/zls', 'karlseguin/http.zig',
        'kristoff-it/zine', 'ziglibs/known-folders', 'Hejsil/zig-clap',
        'ikskuh/zig-network', 'tigerbeetle/tigerbeetle', 'ziglang/zig-bootstrap',
        'kubkon/zig-yaml', 'Vexu/zuri', 'mitchellh/libxev', 'hexops/mach',
        'ziglibs/ini', 'ratfactor/zigish',
    ],
    'sql': [
        'sqlfluff/sqlfluff', 'pingcap/tidb', 'MariaDB/server', 'postgres/postgres',
        'sqlitebrowser/sqlitebrowser', 'supabase/supabase', 'metabase/metabase',
        'dbt-labs/dbt-core', 'apache/spark', 'prisma/prisma', 'hasura/graphql-engine',
        'duckdb/duckdb',
        'liquibase/liquibase', 'flyway/flyway', 'dolthub/dolt',
    ],
    'shell': [
        'ohmyzsh/ohmyzsh', 'nvm-sh/nvm', 'romkatv/powerlevel10k', 'Homebrew/brew',
        'junegunn/fzf', 'dylanaraps/pfetch', 'bats-core/bats-core', 'koalaman/shellcheck',
        'asdf-vm/asdf', 'pi-hole/pi-hole', 'basecamp/omakub', 'tmux-plugins/tpm',
        'sindresorhus/pure', 'rupa/z', 'vundlevim/vundle.vim',
    ],
    'powershell': [
        'PowerShell/PowerShell', 'PowerShell/PSScriptAnalyzer', 'pester/Pester',
        'dfinke/ImportExcel', 'PowerShell/PSReadLine', 'chocolatey/choco',
        'MicrosoftDocs/PowerShell-Docs', 'lukesampson/scoop',
        'PowerShell/PowerShellGet', 'PowerShell/platyPS', 'dsccommunity/DscResource.Test',
        'MicrosoftDocs/windows-powershell-docs', 'pldmgg/misc-powershell',
        'RamblingCookieMonster/PowerShell', 'EvotecIT/PSWriteHTML',
        'PowerShell/DscResources', 'Azure/azure-powershell', 'psake/psake',
    ],
    'html': [
        'h5bp/html5-boilerplate', 'twbs/bootstrap', 'StartBootstrap/startbootstrap-agency',
        'ColorlibHQ/AdminLTE', 'themefisher/meghna-hugo', 'HTML5-Boilerplate/html5-boilerplate',
        'mdn/learning-area', 'w3c/html', 'google/material-design-lite',
        'primer/css', 'PrismJS/prism', 'hakimel/reveal.js',
        'puikinsh/gentelella', 'swagger-api/swagger-ui',
    ],
    'css': [
        'twbs/bootstrap', 'primer/css', 'jgthms/bulma', 'picocss/pico',
        'daneden/animate.css', 'FortAwesome/Font-Awesome', 'tachyons-css/tachyons',
        'ant-design/ant-design', 'nolimits4web/swiper', 'sass/sass-site',
        'h5bp/main.css', 'InfimaUI/infima', 'saadeghi/daisyui', 'openstyles/stylus',
        'necolas/normalize.css', 'milligram/milligram', 'Chalarangelo/mini.css',
        'yeun/open-color', 'basscss/basscss', 'kognise/water.css',
        'oxalorg/sakura', 'dohliam/dropin-minimal-css',
    ],
    'json': [
        'microsoft/vscode', 'DefinitelyTyped/DefinitelyTyped', 'json-schema-org/JSON-Schema-Test-Suite',
        'SchemaStore/schemastore', 'nodejs/node', 'facebook/react', 'sindresorhus/awesome',
        'vercel/next.js',
        'npm/cli', 'oven-sh/bun', 'eslint/eslint', 'babel/babel',
    ],
    'yaml': [
        'kubernetes/kubernetes', 'ansible/ansible', 'helm/charts',
        'docker-library/official-images', 'github/gitignore', 'actions/starter-workflows',
        'prometheus/prometheus', 'grafana/grafana',
        'fluxcd/flux2', 'GoogleCloudPlatform/microservices-demo',
        'docker/awesome-compose', 'openshift/origin',
    ],
    'toml': [
        'rust-lang/cargo', 'rust-lang/rust', 'astral-sh/ruff', 'astral-sh/uv',
        'pypa/pip', 'starship/starship', 'BurntSushi/toml-test', 'tokio-rs/tokio',
    ],
    'markdown': [
        'mdn/content', 'github/docs', 'kubernetes/website', 'microsoft/TypeScript-Website',
        'vercel/next.js', 'nodejs/node', 'rust-lang/reference', 'python/peps',
        'hashicorp/terraform-website', 'gohugoio/hugoDocs', 'apache/spark',
        'elastic/elasticsearch', 'ansible/ansible-documentation', 'docker/docs',
        'w3c/wcag', 'tldr-pages/tldr', 'airbnb/javascript',
        'donnemartin/system-design-primer', 'nextauthjs/next-auth', 'prisma/docs',
        'supabase/supabase', 'grafana/grafana', 'apache/airflow', 'pytorch/tutorials',
        'huggingface/transformers', 'sveltejs/svelte', 'withastro/docs',
        'storybookjs/storybook', 'tauri-apps/tauri-docs', 'railwayapp/docs',
    ],
    'dockerfile': [
        'docker-library/official-images', 'docker-library/postgres', 'docker-library/redis',
        'nodejs/docker-node', 'docker-library/python', 'docker-library/golang',
        'jessfraz/dockerfiles', 'docker-library/mysql', 'docker-library/mongo',
        'docker-library/openjdk', 'docker-library/rabbitmq', 'docker-library/wordpress',
        'docker-library/httpd', 'docker-library/memcached', 'docker-library/ghost',
        'openfaas/faas', 'rancher/rancher',
    ],
    'graphql': [
        'graphql/graphql-js', 'apollographql/apollo-server', 'saleor/saleor',
        'graphile/crystal', '99designs/gqlgen', 'gatsbyjs/gatsby',
        'medusajs/medusa', 'keystonejs/keystone', 'directus/directus',
        'strapi/strapi', 'hasura/graphql-engine', 'shopify/shopify-api-js',
        'artsy/metaphysics', 'graphql-hive/graphql-yoga', 'contentful/contentful.js',
        'wundergraph/wundergraph',
        'dgraph-io/dgraph', 'prisma-labs/graphql-yoga', 'graphql-python/graphene',
    ],
    'hcl': [
        'terraform-aws-modules/terraform-aws-rds', 'terraform-aws-modules/terraform-aws-security-group',
        'cloudposse/terraform-aws-components', 'GoogleCloudPlatform/terraform-google-examples',
        'Azure/terraform', 'hashicorp/terraform', 'terraform-aws-modules/terraform-aws-vpc',
        'terraform-aws-modules/terraform-aws-eks', 'hashicorp/terraform-provider-aws',
        'gruntwork-io/terragrunt', 'hashicorp/packer', 'hashicorp/nomad',
        'terraform-google-modules/terraform-google-network',
    ],
    'dart': [
        'flutter/flutter', 'flutter/samples', 'dart-lang/sdk', 'rrousselGit/riverpod',
        'felangel/bloc', 'jonataslaw/getx', 'flutter/packages', 'dart-lang/http',
        'simplezhli/flutter_deer', 'AppFlowy-IO/AppFlowy',
    ],
    'scala': [
        'apache/spark', 'scala/scala', 'akka/akka', 'playframework/playframework',
        'lampepfl/dotty', 'typelevel/cats', 'zio/zio', 'scalaz/scalaz',
        'apache/kafka', 'twitter/util', 'http4s/http4s', 'softwaremill/tapir',
        'scalatest/scalatest', 'sbt/sbt', 'apache/pekko', 'com-lihaoyi/mill',
    ],
    'perl': [
        'Perl/perl5', 'mojolicious/mojo', 'Perl-Toolchain-Gang/Test-Simple',
        'perl-catalyst/catalyst-runtime', 'houseabsolute/DateTime.pm',
        'libwww-perl/libwww-perl', 'moose/Moose', 'Perl/docs',
        'bioperl/bioperl-live', 'dbsrgits/dbix-class', 'Perl-Critic/Perl-Critic',
        'plack/Plack', 'nigelhorne/Test-Most', 'karenetheridge/Try-Tiny',
        'chorny/Text-CSV_XS', 'timbunce/DBI',
    ],
    'r': [
        'tidyverse/ggplot2', 'tidyverse/dplyr', 'rstudio/shiny', 'tidyverse/tidyr',
        'r-lib/devtools', 'tidyverse/stringr', 'rstudio/rmarkdown', 'Rdatatable/data.table',
        'r-lib/rlang', 'tidymodels/recipes', 'mlr-org/mlr3', 'rstudio/renv',
        'r-lib/cli', 'tidyverse/readr', 'r-lib/vctrs', 'tidyverse/lubridate',
    ],
    'diff': [
        'git/git', 'torvalds/linux', 'python/cpython', 'rust-lang/rust',
        'nodejs/node', 'golang/go', 'llvm/llvm-project', 'openssl/openssl',
    ],
    'plaintext': [
        'first20hours/google-10000-english', 'dwyl/english-words',
        'github/gitignore', 'spdx/license-list-data', 'nodejs/node',
        'python/cpython', 'torvalds/linux', 'git/git',
        'rust-lang/rust', 'golang/go', 'kubernetes/kubernetes', 'facebook/react',
        'django/django', 'rails/rails',
    ],
    'make': [
        'torvalds/linux', 'git/git', 'redis/redis', 'tmux/tmux', 'gohugoio/hugo',
        'buildroot/buildroot', 'openwrt/openwrt', 'gcc-mirror/gcc',
    ],
    'bat': [
        'marlonrichert/zsh-autocomplete', 'chocolatey/choco', 'notepad-plus-plus/notepad-plus-plus',
        'microsoft/vcpkg', 'git-for-windows/build-extra', 'conda/conda',
        'conan-io/conan', 'openresty/openresty', 'microsoft/terminal',
    ],
    'cmake': [
        'Kitware/CMake', 'opencv/opencv', 'assimp/assimp', 'protocolbuffers/protobuf',
        'grpc/grpc', 'nlohmann/json', 'godotengine/godot', 'blender/blender',
    ],
    'objc': ['AFNetworking/AFNetworking', 'SDWebImage/SDWebImage', 'facebook/KVOController', 'BradLarson/GPUImage'],
    'objcpp': [
        'facebook/react-native', 'WebKit/WebKit', 'google/skia',
        'flutter/engine', 'firebase/firebase-ios-sdk',
    ],
    'elisp': [
        'hlissner/doom-emacs', 'magit/magit', 'melpa/melpa', 'purcell/emacs.d',
        'emacs-mirror/emacs', 'jwiegley/use-package', 'magnars/dash.el',
        'company-mode/company-mode', 'emacs-evil/evil',
    ],
    'viml': [
        'tpope/vim-fugitive', 'junegunn/vim-plug', 'preservim/nerdtree', 'morhetz/gruvbox',
        'vim/vim', 'preservim/tagbar', 'neoclide/coc.nvim',
        'dense-analysis/ale', 'sheerun/vim-polyglot',
        'itchyny/lightline.vim',
    ],
    'groovy': [
        'apache/groovy', 'jenkinsci/jenkins', 'spockframework/spock', 'rundeck/rundeck',
        'micronaut-projects/micronaut-core',
    ],
    'asm': [
        'netwide-assembler/nasm', 'x64dbg/x64dbg', 'BLAKE3-team/BLAKE3',
        'reactos/reactos', 'dotnet/runtime', 'openbsd/src',
    ],
    'glsl': [
        'KhronosGroup/glTF-Sample-Viewer', 'patriciogonzalezvivo/glslViewer', 'mrdoob/three.js',
        'libretro/glsl-shaders', 'KhronosGroup/Vulkan-Samples', 'ssloy/tinyrenderer',
        'BabylonJS/Babylon.js', 'pixijs/pixijs',
    ],
    'hlsl': [
        'microsoft/DirectXShaderCompiler', 'microsoft/DirectX-Graphics-Samples', 'gpuopen-librariesandsdks/Cauldron',
        'walbourn/directx-sdk-samples', 'GPUOpen-Effects/FidelityFX-CACAO',
    ],
    'shaderlab': [
        'keijiro/Kino', 'keijiro/PostProcessingUtilities', 'Unity-Technologies/PostProcessing',
        'Unity-Technologies/VolumetricLighting', 'GarrettGunnell/Post-Processing',
    ],
    'plsql': [
        'utPLSQL/utPLSQL', 'oracle-samples/oracle-db-examples', 'dimitri/pgloader', 'dalibo/pgbadger',
        'oracle/oracle-db-examples', 'FSharp-org/plsql-examples',
    ],
    'tsql': ['microsoft/sql-server-samples', 'BrentOzarULTD/SQL-Server-First-Responder-Kit', 'dave-fancher/DataAccessGuide'],
    'tcl': [
        'flightaware/dump1090', 'antirez/redis', 'sqlite/sqlite',
        'tcltk/tcl', 'tcltk/tk', 'tcltk/tcllib',
    ],
    'awk': [
        'onetrueawk/awk', 'step-/JSON.awk', 'soimort/translate-shell', 'freebsd/freebsd',
    ],
    'starlark': [
        'bazelbuild/bazel-skylib', 'bazelbuild/rules_go', 'bazelbuild/rules_docker',
        'bazelbuild/bazel',
    ],
    'hack': [
        'facebook/hhvm', 'hhvm/user-documentation', 'slackhq/hack-json-schema',
        'facebook/flow',
    ],
    'gherkin': [
        'cucumber/cucumber-js', 'cucumber/cucumber-jvm', 'cucumber/cucumber-ruby',
        'behat/behat', 'cucumber/godog',
    ],
    'xslt': [
        'martin-honnen/martin-honnen.github.io', 'highcharts/highcharts', 'apache/fop',
        'docbook/xslt10-stylesheets', 'open-contracting/standard',
        'saxonica/Saxon-HE', 'xmlunit/xmlunit', 'docbook/docbook-xsl',
    ],
    'smarty': [
        'smarty-php/smarty', 'PrestaShop/PrestaShop', 'opencart/opencart',
        'osTicket/osTicket', 'matyhtf/framework',
    ],
    'm4': [
        'autoconf-archive/autoconf-archive', 'bminor/binutils-gdb', 'bminor/glibc',
        'gcc-mirror/gcc', 'westes/flex',
    ],
    'lex': [
        'westes/flex', 'postgres/postgres', 'FreeRDP/FreeRDP',
        'akimd/bison', 'gcc-mirror/gcc',
    ],
    'yacc': [
        'akimd/bison', 'graphviz/graphviz', 'tmux/tmux',
        'postgres/postgres', 'westes/flex', 'gcc-mirror/gcc',
    ],
    'qmake': [
        'qt/qtbase', 'notepad-plus-plus/notepad-plus-plus', 'telegramdesktop/tdesktop',
        'qt/qtdeclarative', 'qt/qttools', 'qt/qtmultimedia',
        'qt/qtcreator', 'qt/qtwebengine',
    ],
}

# Languages whose files are matched by exact filename rather than extension.
FILENAME_MATCH: dict[str, tuple[str, ...]] = {
    'dockerfile': ('dockerfile',),
    'plaintext': ('license', 'authors', 'notice', 'copying', 'contributors', 'changelog'),
    'make': ('makefile', 'gnumakefile'),
    'cmake': ('cmakelists.txt',),
    'starlark': ('build', 'workspace'),
}
