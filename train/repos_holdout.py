"""
A second corpus from repositories the model has never seen.

The main test split holds out *files*, but those files come from the same
repositories the model trained on, which shares house style, identifier
conventions and often whole vendored subtrees. That is a fair split for tracking
training, and an unfair one for comparing against an engine that never saw any of
it.

These repositories must appear nowhere in repos.py, so on this set both engines
are out of distribution and the comparison has no home-field advantage.

That invariant is enforced twice: `holdout_repos()` filters out anything that
also appears in the training list, and test_all.py asserts the filter has nothing
to do. Hand-maintaining two lists is exactly the kind of thing that silently
rots -- an earlier version of this file shared 17 repositories with training,
9 of them in the same language, which would have quietly inflated the headline
number.
"""

from repos import REPOS

HOLDOUT_REPOS: dict[str, list[str]] = {
    'javascript': ['preactjs/preact', 'jquery/jquery', 'date-fns/date-fns',
                   'moment/moment', 'jashkenas/underscore', 'video-dev/hls.js'],
    'typescript': ['type-challenges/type-challenges', 'remix-run/react-router',
                   'immerjs/immer', 'sindresorhus/type-fest'],
    'python': ['psf/black', 'python-poetry/poetry', 'tiangolo/typer', 'sqlalchemy/sqlalchemy'],
    'java': ['iluwatar/java-design-patterns', 'netty/netty', 'google/gson', 'apache/druid'],
    'csharp': ['dotnet/aspnetcore', 'shadowsocks/shadowsocks-windows', 'jbogard/MediatR',
               'dotnet/maui', 'DapperLib/Dapper', 'Fody/Fody'],
    'cpp': ['electron/electron', 'bitcoin/bitcoin', 'pytorch/pytorch', 'apple/swift'],
    'php': ['laravel/laravel', 'bcit-ci/CodeIgniter', 'yiisoft/yii2', 'nextcloud/server'],
    'shell': ['docker/docker-bench-security', 'client9/shlib', 'rbenv/rbenv',
              'ohmybash/oh-my-bash', 'denysdovhan/spaceship-prompt', 'pyenv/pyenv'],
    'c': ['FFmpeg/FFmpeg', 'nginx/nginx', 'php/php-src', 'libgit2/libgit2',
          'obsproject/obs-studio', 'jarun/nnn'],
    'go': ['etcd-io/etcd', 'grpc/grpc-go', 'docker/cli', 'go-gorm/gorm',
           'labstack/echo', 'minio/minio'],
    'html': ['tabler/tabler', 'creativetimofficial/material-dashboard',
             'BlackrockDigital/startbootstrap-sb-admin-2', 'flatlogic/awesome-bootstrap-checkbox'],
    'ruby': ['discourse/discourse', 'mastodon/mastodon', 'rubygems/rubygems',
             'rails/thor', 'ruby/spec', 'jekyll/minima'],
    'css': ['tailwindlabs/tailwindcss', 'ionic-team/ionic-framework', 'uikit/uikit',
            'primefaces/primeng'],
    'markdown': ['vuejs/docs', 'rust-lang/book', 'facebook/docusaurus',
                 'remix-run/remix', 'pnpm/pnpm.io', 'eslint/eslint.org'],
    'rust': ['rust-lang/rustlings', 'denoland/deno', 'alacritty/alacritty', 'nushell/nushell'],
    'kotlin': ['android/nowinandroid', 'square/leakcanary', 'coil-kt/coil', 'cashapp/sqldelight'],
    'swift': ['onevcat/Kingfisher', 'realm/realm-swift', 'airbnb/lottie-ios',
              'SnapKit/SnapKit'],
    'yaml': ['argoproj/argo-cd', 'istio/istio', 'traefik/traefik', 'cilium/cilium'],
    'json': ['microsoft/TypeScript', 'angular/angular', 'denoland/deno', 'expo/expo'],
    'dart': ['flutter/gallery', 'fluttercommunity/plus_plugins', 'flame-engine/flame',
             'localsend/localsend'],
    'scala': ['gatling/gatling', 'twitter/finagle', 'circe/circe', 'scalameta/scalameta'],
    'powershell': ['PowerShell/Win32-OpenSSH', 'microsoft/PowerToys',
                   'adamdriscoll/poshtools', 'PoshCode/PowerShellPracticeAndStyle'],
    'lua': ['neovim/neovim', 'folke/lazy.nvim', 'hrsh7th/nvim-cmp', 'wbthomason/packer.nvim'],
    'perl': ['perl11/cperl', 'rakudo/rakudo', 'shlomif/fc-solve', 'gitpan/Moo',
             'briandfoy/brian-d-foy', 'Perl-Toolchain-Gang/Archive-Tar'],
    'r': ['tidyverse/tibble', 'r-lib/testthat', 'tidyverse/purrr', 'rstudio/gt'],
    # The remaining Sugar High languages, so the head-to-head covers all 33 rather
    # than only the popularity-weighted top 25.
    'zig': ['ratfactor/ziglings', 'buzz-language/buzz', 'natecraddock/zf',
            'zigzap/zap', 'Sobeston/ziglearn', 'MasterQ32/SDL.zig'],
    'hcl': ['terraform-aws-modules/terraform-aws-s3-bucket',
            'terraform-aws-modules/terraform-aws-lambda',
            'cloudposse/terraform-aws-ecs-web-app', 'hashicorp/vault-guides'],
    'toml': ['rust-lang/crates.io', 'helix-editor/helix', 'zellij-org/zellij',
             'rust-lang/rustup', 'sharkdp/hyperfine', 'ajeetdsouza/zoxide'],
    'sql': ['lerocha/chinook-database', 'devrimgunduz/pagila', 'jOOQ/jOOQ',
            'harryho/db-samples', 'datacharmer/test_db', 'morenoh149/postgresDBSamples',
            'ClickHouse/ClickHouse', 'cockroachdb/cockroach'],
    'diff': ['ruby/ruby', 'systemd/systemd', 'mesonbuild/meson', 'strace/strace',
             'util-linux/util-linux', 'htop-dev/htop'],
    'dockerfile': ['docker-library/cassandra', 'docker-library/elasticsearch',
                   'docker-library/tomcat', 'docker-library/haproxy'],
    'graphql': ['github/rest-api-description', 'Shopify/shopify-app-template-node',
                'wpengine/wp-graphql', 'apollographql/apollo-client',
                'gatsbyjs/gatsby-starter-blog', 'vendure-ecommerce/vendure',
                'graphql/graphiql', 'prisma/prisma-examples'],
    'plaintext': ['imsky/wordlists', 'wordnik/wordlist', 'titoBouzout/Dictionaries',
                  'stopwords-iso/stopwords-en', 'dwyl/quotes',
                  'arstgit/high-frequency-vocabulary'],
}


_TRAIN = {r for v in REPOS.values() for r in v}


def holdout_repos() -> dict[str, list[str]]:
    """Holdout list with any repository that also appears in training removed.

    The filter is the enforcement, not a convenience: a repository shared with
    training is not held out, whatever list it is written in.
    """
    return {lang: [r for r in repos if r not in _TRAIN]
            for lang, repos in HOLDOUT_REPOS.items()}


def overlap() -> dict[str, list[str]]:
    """Repositories present in both lists, for the test that asserts there are none."""
    return {lang: [r for r in repos if r in _TRAIN]
            for lang, repos in HOLDOUT_REPOS.items()
            if any(r in _TRAIN for r in repos)}
