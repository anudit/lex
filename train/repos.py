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
    ],
    'typescript': [
        'microsoft/vscode', 'nestjs/nest', 'trpc/trpc', 'pmndrs/zustand',
        'colinhacks/zod', 'TanStack/query', 'vitejs/vite', 'excalidraw/excalidraw',
    ],
    'python': [
        'pallets/flask', 'psf/requests', 'django/django', 'pandas-dev/pandas',
        'encode/httpx', 'pydantic/pydantic', 'fastapi/fastapi', 'scrapy/scrapy',
    ],
    'rust': [
        'BurntSushi/ripgrep', 'sharkdp/bat', 'tokio-rs/tokio', 'serde-rs/serde',
        'clap-rs/clap', 'rust-lang/regex', 'starship/starship', 'sharkdp/fd',
    ],
    'go': [
        'gin-gonic/gin', 'spf13/cobra', 'sirupsen/logrus', 'gorilla/mux',
        'go-chi/chi', 'urfave/cli', 'stretchr/testify', 'hashicorp/consul',
    ],
    'c': [
        'curl/curl', 'redis/redis', 'git/git', 'jqlang/jq',
        'libuv/libuv', 'nothings/stb', 'antirez/sds', 'DaveGamble/cJSON',
    ],
    'cpp': [
        'nlohmann/json', 'fmtlib/fmt', 'google/googletest', 'gabime/spdlog',
        'catchorg/Catch2', 'protocolbuffers/protobuf', 'opencv/opencv', 'ocornut/imgui',
    ],
    'csharp': [
        'dotnet/runtime', 'AutoMapper/AutoMapper', 'JamesNK/Newtonsoft.Json',
        'serilog/serilog', 'App-vNext/Polly', 'restsharp/RestSharp',
        'quartznet/quartznet', 'MassTransit/MassTransit',
    ],
    'java': [
        'google/guava', 'square/retrofit', 'square/okhttp', 'ReactiveX/RxJava',
        'alibaba/fastjson', 'spring-projects/spring-framework', 'apache/commons-lang',
        'junit-team/junit5', 'spring-projects/spring-boot', 'elastic/elasticsearch',
        'apache/dubbo', 'mybatis/mybatis-3', 'google/error-prone', 'apache/flink',
        'apache/lucene', 'quarkusio/quarkus',
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
    ],
    'php': [
        'laravel/framework', 'symfony/symfony', 'guzzle/guzzle', 'composer/composer',
        'phpunit/phpunit', 'monolog/monolog', 'slimphp/Slim', 'doctrine/orm',
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
    ],
    'shell': [
        'ohmyzsh/ohmyzsh', 'nvm-sh/nvm', 'romkatv/powerlevel10k', 'Homebrew/brew',
        'junegunn/fzf', 'dylanaraps/pfetch', 'bats-core/bats-core', 'koalaman/shellcheck',
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
    ],
    'yaml': [
        'kubernetes/kubernetes', 'ansible/ansible', 'helm/charts',
        'docker-library/official-images', 'github/gitignore', 'actions/starter-workflows',
        'prometheus/prometheus', 'grafana/grafana',
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
    ],
    'graphql': [
        'graphql/graphql-js', 'apollographql/apollo-server', 'saleor/saleor',
        'graphile/crystal', '99designs/gqlgen', 'gatsbyjs/gatsby',
        'medusajs/medusa', 'keystonejs/keystone', 'directus/directus',
        'strapi/strapi', 'hasura/graphql-engine', 'shopify/shopify-api-js',
        'artsy/metaphysics', 'graphql-hive/graphql-yoga', 'contentful/contentful.js',
        'wundergraph/wundergraph',
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
    ],
}

# Languages whose files are matched by exact filename rather than extension.
FILENAME_MATCH: dict[str, tuple[str, ...]] = {
    'dockerfile': ('dockerfile',),
    'plaintext': ('license', 'authors', 'notice', 'copying', 'contributors', 'changelog'),
}
