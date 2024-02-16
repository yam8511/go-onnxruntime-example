package utils

import "strings"

func MetadataToNames(_names string) []string {
	names := []string{}
	_names_2 := strings.Split(strings.TrimSuffix(strings.TrimPrefix(_names, "{"), "}"), ",")
	for _, name := range _names_2 {
		name2 := strings.Split(name, ":")
		if len(name2) > 1 {
			names = append(names, strings.Trim(strings.TrimSpace(name2[1]), `'`))
		}
	}
	return names
}
