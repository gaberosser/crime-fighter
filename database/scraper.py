import urllib2
import re
import shapefile


PAGES = (
    "http://content.met.police.uk/Page/TeamFinder?scope_id=1257246763604",
)

# if __name__ == "__main__":

    # with open('/home/gabriel/source1.html') as f:
    #     data = f.read()


def extract_safer_neighbourhoods(data):
    """
    data is a string containing the source from a Safer Neighbourhood website, such as this:
    http://content.met.police.uk/Page/TeamFinder?scope_id=1257246763604
    """
    # split into lines
    res = re.sub(r'\n+', '\n', data.replace('\r', '').replace('\t', '')).split('\n')
    a = []
    for r in res:
        t = re.sub(r'^ *', '', r)
        if t:
            a.append(t)

    reader = (r for r in a)

    out = []
    i = 0
    # iterate over lines looking for the starting keyword
    try:
        while True:
            r = reader.next()
            s = re.search(r"arrTeams\[[0-9]*\]\[0\] = '(?P<code>\w*)';", r)
            if s:
                code = s.group('code')
                # read the polygon
                lonlat = []
                i += 1
                r = reader.next()
                # run until the end
                while True:
                    # look for end
                    s = re.search(r"arrTeams\[[0-9]*\]\[2\] = '<strong>(?P<name>[a-zA-Z ]+)<", r)
                    if s:
                        name = s.group('name')
                        break
                    else:
                        s = re.search(r"new google.maps.LatLng\((?P<lat>[-0-9.]*), (?P<lon>[-0-9.]*)\),", r)
                        if s:
                            lat = float(s.group('lat'))
                            lon = float(s.group('lon'))
                            lonlat.append((lon, lat))
                    r = reader.next()
                out.append({
                    'name': name,
                    'code': code,
                    'lonlat': lonlat
                })
    except StopIteration:
        print "Read %d units" % i

    return out


def get_source(url):
    response = urllib2.urlopen(url)
    return response.read()


def get_sn_links():
    base_url = "http://content.met.police.uk/Site/YourBorough"
    out = []
    data = get_source(base_url).splitlines()
    patt = r'a href="(?P<url>/Page/TeamFinder\?scope_id=[0-9]*)"'
    for r in data:
        s = re.search(patt, r)
        if s:
            out.append('http://content.met.police.uk/' + s.group('url'))
    return out


def write_to_shapefile(data, outfile):
    w = shapefile.Writer(shapefile.POLYGON)
    w.autoBalance = 1
    w.field('name', fieldType='C')
    w.field('code', fieldType='C')
    for x in data:
        w.record(name=x['name'], code=x['code'])
        if x['lonlat']:
            w.poly(parts=[x['lonlat']])
    w.save(outfile)


if __name__ == "__main__":

    data = []
    urls = get_sn_links()
    for u in urls:
        src = get_source(u)
        data.extend(extract_safer_neighbourhoods(src))
    write_to_shapefile(data, 'safer_neighbourhoods')