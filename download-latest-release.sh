# download latest RELEASE of the software
# copied from https://geraldonit.com/2019/01/15/how-to-download-the-latest-github-repo-release-via-command-line/

LOCATION=$(curl -s https://api.github.com/repos/PhilipSargent/h2-in-pipes/releases/latest \
| grep "tag_name" \
| awk '{print "https://github.com/PhilipSargent/h2-in-pipes/archive/" substr($2, 2, length($2)-3) ".zip"}')\
> ; curl -L -o h2-in-pipes.zip $LOCATION

 