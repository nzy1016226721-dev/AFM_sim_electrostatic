"""Local-only visualization/sanity checks AFTER MPI completion and download."""
import argparse
import json
from pathlib import Path
from local_visualization import require_local


def main(argv=None):
    require_local()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["list","coordinates","planes","lines","profile","draw","sanity"])
    parser.add_argument("files",nargs="*")
    parser.add_argument("--directory",help="Downloaded NPY directory")
    parser.add_argument("--select",default="",help="Zero-based inclusive 0-5,8 or 0:20:2 selection")
    parser.add_argument("--bounds",nargs=6,type=float,help="FIRST/LAST retained x,y,z node coordinates in nm (not bin edges)")
    parser.add_argument("--config",help="Exact JSON used by the completed MPI run, for coordinate reconstruction")
    parser.add_argument("--center",nargs=3,type=float,help="Saved case's movement center xyz in physical nm")
    parser.add_argument("--axis",choices=["x","y","z"],default="z")
    parser.add_argument("--plane",choices=["xy","xz","yz"],default="xz")
    parser.add_argument("--at",default="0",help="Comma-separated normal positions in nm")
    parser.add_argument("--first",default="0",help="First other-axis nm positions for axis lines")
    parser.add_argument("--second",default="0",help="Second other-axis nm positions for axis lines")
    parser.add_argument("--start",nargs=3,type=float)
    parser.add_argument("--end",nargs=3,type=float)
    parser.add_argument("--samples",type=int,default=500)
    parser.add_argument("--title")
    parser.add_argument("--limits",nargs=2,type=float)
    parser.add_argument("--out",default="outputs/local_plots",help="Local derived figures only; input NPYs are read-only")
    parser.add_argument("--show",action="store_true")
    parser.add_argument("--electric-field",action="store_true",help="In-plane E arrows, not a full-field convergence test")
    args = parser.parse_args(argv)
    from local_visualization.bulk_potential_plotter import npy_file_sorter,list_to_array_float,plane_plotter,line_plotter
    from local_visualization.data import PotentialData,select_indices
    paths = list(map(Path,args.files))
    if args.directory:
        available = npy_file_sorter(args.directory)
        if args.command == "list":
            for i,path in enumerate(available):
                print(f"{i}: {path.name}")
            return 0
        paths += [available[i] for i in select_indices(args.select,len(available))]
    if not paths:
        parser.error("Supply downloaded .npy files or --directory")
    if args.command == "coordinates":
        if args.config is None:
            parser.error("coordinates requires the exact --config used by the MPI job")
        from local_visualization.coordinates import reconstruct_coordinates
        for path in paths:
            print(reconstruct_coordinates(path,args.config,args.center))
        return 0
    if args.command == "sanity":
        records = []
        for path in paths:
            with PotentialData(path,args.bounds) as data:
                records.append(data.sanity())
        print(json.dumps(records,indent=2))
        return 0 if all(record["nonfinite_count"]==0 for record in records) else 2
    if args.command == "planes":
        saved = plane_plotter(paths,args.plane,list_to_array_float(args.at),bounds_nm=args.bounds,
                              title=args.title,output_dir=args.out,show=args.show,electric_field=args.electric_field)
    elif args.command == "lines":
        saved = line_plotter(paths,args.axis,[list_to_array_float(args.first),list_to_array_float(args.second)],
                             bounds_nm=args.bounds,title=args.title,limits=args.limits,output_dir=args.out,show=args.show)
    else:
        from local_visualization.line_profile_plotter import line_profile_plotter,drawn_profile
        saved = []
        for path in paths:
            if args.command == "draw":
                saved.append(drawn_profile(path,args.plane,list_to_array_float(args.at)[0],bounds_nm=args.bounds,samples=args.samples,output_dir=args.out))
            elif args.command == "profile":
                if args.start is None or args.end is None:
                    parser.error("profile requires --start X Y Z and --end X Y Z")
                saved.append(line_profile_plotter(path,args.start,args.end,bounds_nm=args.bounds,samples=args.samples,title=args.title,output_dir=args.out,show=args.show))
            else:
                parser.error("Use list with --directory")
    for path in saved:
        if path:
            print(f"Saved local figure: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
